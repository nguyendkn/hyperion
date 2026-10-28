//! Engine state machine — the lifecycle the runtime moves through from boot
//! to shutdown. Lives here (not in the HAL) because the transitions are
//! device-agnostic: they describe *engine* state, not *device* state.
//!
//! Transitions are driven by the CLI and the API layer. They are deliberately
//! restrictive: you cannot prefill before the model is loaded, cannot load a
//! second model while one is running, etc. This makes the engine's invariants
//! checkable by inspection and prevents the half-initialised panics the
//! architecture doc's "runtime dynamic triage" flow is designed to avoid.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::Result;
use thiserror::Error;

/// The states an engine can be in. Stored as `u8` so a single atomic carries
/// the current state lock-free; reads from the API hot path never contend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineStatus {
    /// Freshly constructed. No device probed, no model loaded.
    Uninitialized = 0,
    /// HAL backend selected and device context allocated. Triage passed.
    Ready = 1,
    /// Weights are in device memory; KV cache allocated. Servable.
    Loaded = 2,
    /// A request (prefill or decode) is in flight on the device.
    Serving = 3,
    /// Draining in-flight work before shutdown. New requests rejected.
    Draining = 4,
    /// Final state after `shutdown`. No further transitions allowed.
    Stopped = 5,
}

impl EngineStatus {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Uninitialized),
            1 => Some(Self::Ready),
            2 => Some(Self::Loaded),
            3 => Some(Self::Serving),
            4 => Some(Self::Draining),
            5 => Some(Self::Stopped),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("illegal transition: {from:?} -> {to:?}")]
    IllegalTransition { from: EngineStatus, to: EngineStatus },
    #[error("engine already terminal: {0:?}")]
    Terminal(EngineStatus),
}

/// Allowed-transition table. Encoded as a match so the compiler tells us when
/// a new state variant is added without a transition rule — exhaustiveness is
/// the safety net.
fn can_transition(from: EngineStatus, to: EngineStatus) -> bool {
    use EngineStatus::*;
    matches!(
        (from, to),
        (Uninitialized, Ready)
            | (Ready, Loaded)
            | (Loaded, Serving)
            | (Serving, Loaded)
            | (Loaded | Serving | Ready, Draining)
            | (Draining, Stopped)
            | (Loaded | Ready, Stopped)
            | (Stopped, Stopped)
    )
}

/// Lock-free engine state holder. The `AtomicU8` is the source of truth for
/// "what state are we in"; the scheduler and API layer observe it before
/// acting. A closed set of states + a CAS loop make transitions safe under
/// concurrent API requests without a heavyweight mutex on the hot path.
#[derive(Debug)]
pub struct EngineState {
    status: AtomicU8,
}

impl Default for EngineState {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineState {
    pub fn new() -> Self {
        Self {
            status: AtomicU8::new(EngineStatus::Uninitialized as u8),
        }
    }

    pub fn current(&self) -> EngineStatus {
        EngineStatus::from_u8(self.status.load(Ordering::Acquire))
            .expect("invalid state byte — memory corruption")
    }

    /// Move to `to` only if the transition is legal AND happens atomically
    /// vs concurrent movers. Returns the previous state on success.
    pub fn transition(&self, to: EngineStatus) -> Result<EngineStatus> {
        loop {
            let from = self.current();
            if from == EngineStatus::Stopped {
                // Allow idempotent Stopped->Stopped for clean shutdown racing.
                if to == EngineStatus::Stopped {
                    return Ok(from);
                }
                return Err(StateError::Terminal(from).into());
            }
            if !can_transition(from, to) {
                return Err(StateError::IllegalTransition { from, to }.into());
            }
            if self
                .status
                .compare_exchange(from as u8, to as u8, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(from);
            }
            // CAS lost — another thread moved first; retry from the new state.
        }
    }
}

/// Thin wrapper exposing the lifecycle transitions the engine expects. The
/// API layer holds one of these and consults `status()` before acting on a
/// request; the scheduler feeds `begin_serving`/`end_serving` around each
/// batch. The backend handle itself lives in the scheduler — this struct is
/// the *state*, not the engine.
///
/// `Clone` is sound only because every field is behind an `Arc` — cloning
/// produces another handle to the *same* `EngineState`, so all clones agree
/// on the current status. If a non-`Arc` field is ever added, `Clone` here
/// would silently let clones diverge state; revisit then.
#[derive(Clone)]
pub struct StateMachine {
    pub state: Arc<EngineState>,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            state: Arc::new(EngineState::new()),
        }
    }

    pub fn status(&self) -> EngineStatus {
        self.state.current()
    }

    pub fn mark_ready(&self) -> Result<()> {
        self.state.transition(EngineStatus::Ready).map(|_| ())
    }

    pub fn mark_loaded(&self) -> Result<()> {
        self.state.transition(EngineStatus::Loaded).map(|_| ())
    }

    pub fn begin_serving(&self) -> Result<()> {
        self.state.transition(EngineStatus::Serving).map(|_| ())
    }

    pub fn end_serving(&self) -> Result<()> {
        self.state.transition(EngineStatus::Loaded).map(|_| ())
    }

    pub fn drain(&self) -> Result<()> {
        self.state.transition(EngineStatus::Draining).map(|_| ())
    }

    /// Shutdown-friendly drain: move to `Draining` from `Ready`, `Loaded` or
    /// `Serving`. Already being shut down (`Draining`, or `Stopped`, which is
    /// terminal) counts as success so a racing supervisor is harmless. Any
    /// other failure, notably `Uninitialized`, is returned to the caller.
    pub fn drain_for_shutdown(&self) -> Result<()> {
        match self.drain() {
            Ok(()) => Ok(()),
            Err(e) => match e.downcast_ref::<StateError>() {
                Some(StateError::IllegalTransition { from: EngineStatus::Draining, .. })
                | Some(StateError::Terminal(_)) => Ok(()),
                _ => Err(e),
            },
        }
    }

    pub fn shutdown(&self) -> Result<()> {
        self.state.transition(EngineStatus::Stopped).map(|_| ())
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use EngineStatus::*;

    const ALL: [EngineStatus; 6] = [Uninitialized, Ready, Loaded, Serving, Draining, Stopped];

    fn at(status: EngineStatus) -> EngineState {
        EngineState { status: AtomicU8::new(status as u8) }
    }

    #[test]
    fn transition_table_is_exactly_the_documented_set() {
        let allowed = [
            (Uninitialized, Ready),
            (Ready, Loaded),
            (Loaded, Serving),
            (Serving, Loaded),
            (Loaded, Draining),
            (Serving, Draining),
            (Ready, Draining),
            (Draining, Stopped),
            (Loaded, Stopped),
            (Ready, Stopped),
            (Stopped, Stopped),
        ];
        for from in ALL {
            for to in ALL {
                assert_eq!(
                    can_transition(from, to),
                    allowed.contains(&(from, to)),
                    "{from:?} -> {to:?}"
                );
            }
        }
    }

    #[test]
    fn drain_for_shutdown_from_every_state() {
        for (from, ok, after) in [
            (Uninitialized, false, Uninitialized),
            (Ready, true, Draining),
            (Loaded, true, Draining),
            (Serving, true, Draining),
            (Draining, true, Draining),
            (Stopped, true, Stopped),
        ] {
            let sm = StateMachine { state: Arc::new(at(from)) };
            assert_eq!(sm.drain_for_shutdown().is_ok(), ok, "from {from:?}");
            assert_eq!(sm.status(), after, "from {from:?}");
        }
    }

    #[test]
    fn stopped_is_terminal_except_idempotent_stop() {
        let st = at(Stopped);
        assert!(st.transition(Stopped).is_ok());
        for to in [Ready, Loaded, Serving, Draining] {
            assert!(st.transition(to).is_err(), "Stopped -> {to:?}");
        }
    }
}
