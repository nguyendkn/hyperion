//! The order of `submit_step` and `take_output` (ADR 0006): steps are submitted in plan order, their outputs are taken in the same order, and the
//! host may be at most `max_in_flight` steps ahead (the output of a step can be collected one step late, so two are in flight at most). A backend
//! keeps one ledger; it refuses a submit past the limit and a take out of order, with typed errors. Pure bookkeeping, tested on any host.

use std::collections::VecDeque;

use crate::contract::HalError;

#[derive(Debug)]
pub struct StepLedger {
    max_in_flight: usize,
    in_flight: VecDeque<u64>,
    next_ticket: u64,
}

impl StepLedger {
    pub fn new(max_in_flight: usize) -> Self {
        Self { max_in_flight, in_flight: VecDeque::new(), next_ticket: 0 }
    }

    /// Register a step; its ticket number. Refused while `max_in_flight` steps have not been taken.
    pub fn submit(&mut self) -> Result<u64, HalError> {
        if self.in_flight.len() >= self.max_in_flight {
            return Err(HalError::Invalid(format!("{} steps are in flight: take an output before submitting another", self.in_flight.len())));
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.in_flight.push_back(ticket);
        Ok(ticket)
    }

    /// Release the oldest step. Only the oldest may be taken; a ticket taken twice or out of order is refused and leaves the ledger as it was.
    pub fn take(&mut self, ticket: u64) -> Result<(), HalError> {
        match self.in_flight.front() {
            Some(&oldest) if oldest == ticket => {
                self.in_flight.pop_front();
                Ok(())
            }
            Some(&oldest) => Err(HalError::Invalid(format!("step {ticket} taken before step {oldest}: outputs are taken in submission order"))),
            None => Err(HalError::Invalid(format!("step {ticket} is not in flight"))),
        }
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_are_taken_in_submission_order_with_one_step_of_lag() {
        let mut ledger = StepLedger::new(2);
        let (a, b) = (ledger.submit().unwrap(), ledger.submit().unwrap());
        assert_eq!((a, b, ledger.in_flight()), (0, 1, 2));
        assert!(ledger.submit().is_err(), "a third step waits for an output");
        assert!(ledger.take(b).is_err(), "step 1 cannot be taken before step 0");
        assert_eq!(ledger.in_flight(), 2, "a refused take changes nothing");
        ledger.take(a).unwrap();
        let c = ledger.submit().unwrap();
        ledger.take(b).unwrap();
        ledger.take(c).unwrap();
        assert!(ledger.take(c).is_err(), "a ticket cannot be taken twice");
        assert_eq!(ledger.in_flight(), 0);
    }

    #[test]
    fn a_ledger_of_one_completes_each_step_before_the_next() {
        let mut ledger = StepLedger::new(1);
        let t = ledger.submit().unwrap();
        assert!(ledger.submit().is_err());
        ledger.take(t).unwrap();
        assert_eq!(ledger.submit().unwrap(), 1);
    }
}
