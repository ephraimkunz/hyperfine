//! Cache and branch-miss counters on Apple Silicon.
//!
//! `proc_pid_rusage` only reports cycles and instructions. The remaining
//! counters are read by having kperf sample the child's threads, which needs
//! root and briefly takes over the system-wide kdebug trace buffer.

use std::io::{self, PipeWriter, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{self, Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use darwin_kperf::event::Event;
use darwin_kperf::{ProcessCounters, Sampler};

use super::unix_timer::CPUTimer;
use super::wall_clock_timer::WallClockTimer;
use super::{discard, macos_counters};
use crate::benchmark::measurement::{HardwareCounters, Measurement};

const EVENTS: [Event; 3] = [
    Event::BranchMispredNonspec,
    Event::L1DCacheMissLdNonspec,
    Event::L1ICacheMissDemand,
];

/// Sampling interrupts the benchmarked process, so this trades counter
/// coverage at the start and end of a run against wall-clock disturbance.
const SAMPLE_PERIOD: Duration = Duration::from_millis(1);

const DRAIN_INTERVAL: Duration = Duration::from_millis(10);

/// Root privileges do not change while hyperfine runs, so one failed attempt
/// is enough to stop trying for every remaining run.
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);

pub fn sampler() -> Option<Sampler> {
    if UNAVAILABLE.load(Ordering::Relaxed) {
        return None;
    }
    let sampler = Sampler::new().ok();
    if sampler.is_none() {
        UNAVAILABLE.store(true, Ordering::Relaxed);
    }
    sampler
}

/// A forked child that is blocked just before it execs the command.
///
/// kperf only counts what happens while it is sampling, so the command must
/// not start running until sampling has been set up for its pid.
struct HeldChild {
    pid: i32,
    release: PipeWriter,
    spawning: JoinHandle<io::Result<Child>>,
}

impl HeldChild {
    fn spawn(mut command: Command) -> io::Result<Self> {
        let (mut pid_rx, mut pid_tx) = io::pipe()?;
        let (mut release_rx, release_tx) = io::pipe()?;
        let release_tx_fd = release_tx.as_raw_fd();

        // SAFETY: only close, getpid, write and read run between fork and exec.
        unsafe {
            command.pre_exec(move || {
                // The fork inherited the write end. While the child holds it, a
                // parent that fails before releasing would never produce EOF here.
                libc::close(release_tx_fd);
                pid_tx.write_all(&process::id().to_ne_bytes())?;
                release_rx.read_exact(&mut [0])
            });
        }

        // `spawn` does not return until the child has exec'd, so it cannot run
        // on the thread that releases the child. Moving the command also drops
        // its pipe ends if spawning fails, which ends the read below.
        let spawning = thread::spawn(move || command.spawn());

        let mut pid = [0; 4];
        if pid_rx.read_exact(&mut pid).is_err() {
            return Err(match spawning.join() {
                Ok(Err(error)) => error,
                _ => io::Error::other("child process did not reach exec"),
            });
        }

        Ok(Self {
            pid: i32::from_ne_bytes(pid),
            release: release_tx,
            spawning,
        })
    }

    fn release(mut self) -> io::Result<Child> {
        self.release.write_all(&[0])?;
        self.spawning
            .join()
            .map_err(|_| io::Error::other("spawning the child process panicked"))?
    }
}

fn sampled_counters(counters: &ProcessCounters<3>) -> Option<(u64, u64)> {
    // No thread sampled twice means the run was too short to measure, which
    // is different from a run that had no misses.
    if counters.threads.is_empty() {
        return None;
    }
    let [branch_misses, data_cache_misses, instruction_cache_misses] = counters.totals;
    Some((
        branch_misses,
        data_cache_misses.saturating_add(instruction_cache_misses),
    ))
}

pub fn execute_and_measure(command: Command, sampler: &Sampler) -> Result<Measurement> {
    let cpu_timer = CPUTimer::start();
    let held = HeldChild::spawn(command)?;

    // A failure here must not skip `release`, or the child would stay blocked.
    let pid_sampler = sampler
        .pid(EVENTS, held.pid)
        .ok()
        .map(|pid_sampler| pid_sampler.with_sample_period(SAMPLE_PERIOD))
        .and_then(|mut pid_sampler| pid_sampler.start().ok().map(|()| pid_sampler));

    let wallclock_timer = WallClockTimer::start();
    let mut child = held.release()?;

    let finished = AtomicBool::new(false);
    let (time_wall_clock, exited, sampled) = thread::scope(|scope| {
        // The kernel trace buffer overflows unless it is emptied while the
        // command runs, and this thread is blocked waiting for the child.
        let drainer = pid_sampler.map(|mut pid_sampler| {
            let finished = &finished;
            scope.spawn(move || {
                while !finished.load(Ordering::Acquire) {
                    thread::park_timeout(DRAIN_INTERVAL);
                    let _ = pid_sampler.drain();
                }
                pid_sampler.stop().ok()
            })
        });

        if let Some(output) = child.stdout.take() {
            discard(output);
        }
        let exited = macos_counters::wait(&mut child).is_ok();
        let time_wall_clock = exited.then(|| wallclock_timer.stop());

        finished.store(true, Ordering::Release);
        let sampled = drainer.and_then(|drainer| {
            drainer.thread().unpark();
            drainer.join().ok().flatten()
        });

        (time_wall_clock, exited, sampled)
    });

    let mut hardware_counters = if exited {
        macos_counters::read(child.id())
    } else {
        HardwareCounters::default()
    };
    if let Some((branch_misses, cache_misses)) = sampled
        .filter(|_| exited)
        .as_ref()
        .and_then(sampled_counters)
    {
        hardware_counters.branch_misses = Some(branch_misses);
        hardware_counters.cache_misses = Some(cache_misses);
    }

    let (time_user, time_system, memory_peak_resident, exit_status) = cpu_timer.stop(child)?;
    let time_wall_clock = time_wall_clock.unwrap_or_else(|| wallclock_timer.stop());

    Ok(Measurement {
        time_wall_clock,
        time_cpu: time_user + time_system,
        time_user,
        time_system,
        memory_peak_resident,
        hardware_counters,
        exit_status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_child_runs_only_after_release() {
        let marker = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(marker.path()).unwrap();

        let mut command = Command::new("touch");
        command.arg(marker.path());
        let held = HeldChild::spawn(command).unwrap();
        assert!(held.pid > 0);

        thread::sleep(Duration::from_millis(100));
        assert!(!marker.path().exists());

        let mut child = held.release().unwrap();
        assert!(child.wait().unwrap().success());
        assert!(marker.path().exists());
        std::fs::remove_file(marker.path()).unwrap();
    }

    #[test]
    fn held_child_pid_matches_spawned_child() {
        let held = HeldChild::spawn(Command::new("true")).unwrap();
        let pid = held.pid;
        let mut child = held.release().unwrap();
        assert_eq!(child.id() as i32, pid);
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn missing_command_is_reported_on_release() {
        let held = HeldChild::spawn(Command::new("hyperfine-no-such-command")).unwrap();
        let error = held.release().unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn dropping_a_held_child_unblocks_it() {
        let held = HeldChild::spawn(Command::new("true")).unwrap();
        let HeldChild {
            release, spawning, ..
        } = held;
        drop(release);
        assert!(spawning.join().unwrap().is_err());
    }

    #[test]
    fn runs_without_samples_report_no_counters() {
        let counters = ProcessCounters {
            totals: [0; 3],
            threads: Vec::new(),
        };
        assert_eq!(sampled_counters(&counters), None);
    }
}
