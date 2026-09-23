// Scratch is allocated once with the TP group, not on the submission path.
pub(crate) fn reserve_all<T: Copy, E>(
    tickets: &mut [Option<T>],
    mut preflight: impl FnMut(usize) -> Result<T, E>,
    mut reserve: impl FnMut(usize, T) -> Result<(), E>,
    mut cancel: impl FnMut(usize),
) -> Result<(), E> {
    for (rank, slot) in tickets.iter_mut().enumerate() {
        *slot = Some(preflight(rank)?);
    }
    for (rank, ticket) in tickets.iter().enumerate() {
        if let Err(error) = reserve(rank, ticket.expect("all ranks preflighted")) {
            // The failed call may already have reserved AQL positions. A prior
            // rank may be prepared; no rank may silently continue this collective.
            for rank in 0..tickets.len() {
                cancel(rank);
            }
            return Err(error);
        }
    }
    Ok(())
}

pub(crate) fn finish_all<E>(
    ranks: usize,
    mut finish: impl FnMut(usize) -> Result<(), E>,
) -> Result<(), E> {
    let mut first_error = None;
    for rank in 0..ranks {
        if let Err(error) = finish(rank) {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
#[path = "../device/kernarg_retirement.rs"]
mod retirement;

#[cfg(test)]
mod tests {
    use super::{
        finish_all, reserve_all,
        retirement::{Error, KernargRetirement},
    };
    use std::cell::{Cell, RefCell};

    #[test]
    fn teardown_attempts_every_rank_before_returning_first_failure() {
        let events = RefCell::new(Vec::new());
        let result = finish_all(4, |rank| {
            events.borrow_mut().push(("finish", rank));
            if rank == 1 || rank == 3 {
                Err(rank)
            } else {
                Ok(())
            }
        });
        if result.is_err() {
            for rank in 0..4 {
                events.borrow_mut().push(("cancel", rank));
            }
        }
        events.borrow_mut().push(("return", 0));
        assert_eq!(result, Err(1));
        assert_eq!(
            *events.borrow(),
            [
                ("finish", 0),
                ("finish", 1),
                ("finish", 2),
                ("finish", 3),
                ("cancel", 0),
                ("cancel", 1),
                ("cancel", 2),
                ("cancel", 3),
                ("return", 0),
            ]
        );
    }

    #[test]
    fn immediate_batch_keeps_per_packet_doorbells_and_failure_never_retires() {
        use super::retirement::{ring_on_commit, ring_on_dispatch};
        assert!(ring_on_dispatch(false, || panic!(
            "ordinary dispatch must not read reservation mode"
        )));
        let ranks: [_; 3] = std::array::from_fn(|_| KernargRetirement::new());
        let mut tickets = [None; 3];
        reserve_all(
            &mut tickets,
            |rank| ranks[rank].preflight(3, 4, || 0),
            |rank, ticket| ranks[rank].reserve_admitted(ticket).map(|_| ()),
            |_| panic!("valid admission"),
        )
        .unwrap();
        let mut doorbells = Vec::new();
        for packet in 0..3 {
            if ring_on_dispatch(true, || true) {
                doorbells.push((0, packet));
            }
            assert!(!ring_on_dispatch(true, || false));
        }
        assert_eq!(doorbells, [(0, 0), (0, 1), (0, 2)]);
        assert!(!ring_on_commit(true));
        assert!(ring_on_commit(false));
        // Rank 1 fails after rank 0 has actually been signaled, before rank 2 emits.
        for rank in &ranks {
            rank.cancel();
        }
        for rank in &ranks {
            assert_eq!(rank.complete(3, 0), Err(Error::Pending));
            assert_eq!(rank.reserve(1, 4), Err(Error::Pending));
        }
    }

    #[test]
    fn exhausted_rank_leaves_every_queue_unmodified() {
        let ranks: [_; 3] = std::array::from_fn(|_| KernargRetirement::new());
        for rank in &ranks {
            rank.reserve(4, 4).unwrap();
            rank.publish(4).unwrap();
        }
        let mut tickets = [None; 3];
        let result = reserve_all(
            &mut tickets,
            |rank| ranks[rank].preflight(1, 4, || if rank == 2 { 1 } else { 0 }),
            |_, _| -> Result<(), Error> { panic!("preflight must finish before any reservation") },
            |_| panic!("preflight failure must not cancel existing work"),
        );
        assert_eq!(result, Err(Error::Incomplete(1)));
        for rank in &ranks {
            assert_eq!(rank.completion_ticket(), Ok(4));
            assert_eq!(rank.reserve(1, 4), Err(Error::Capacity));
        }
    }

    #[test]
    fn successful_collective_preflights_all_before_reserving_any() {
        let ranks: [_; 3] = std::array::from_fn(|_| KernargRetirement::new());
        let mut tickets = [None; 3];
        let events = RefCell::new(Vec::new());
        reserve_all(
            &mut tickets,
            |rank| {
                events.borrow_mut().push(('P', rank));
                ranks[rank].preflight(4, 4, || 0)
            },
            |rank, ticket| {
                events.borrow_mut().push(('R', rank));
                ranks[rank].reserve_admitted(ticket).map(|_| ())
            },
            |_| panic!("valid collective"),
        )
        .unwrap();
        assert_eq!(
            *events.borrow(),
            [('P', 0), ('P', 1), ('P', 2), ('R', 0), ('R', 1), ('R', 2)]
        );
        for rank in &ranks {
            rank.publish(4).unwrap();
        }
        reserve_all(
            &mut tickets,
            |rank| ranks[rank].preflight(4, 4, || 0),
            |rank, ticket| {
                ranks[rank]
                    .reserve_admitted(ticket)
                    .map(|base| assert_eq!(base, 4))
            },
            |_| panic!("completed collective wrap"),
        )
        .unwrap();
        for rank in &ranks {
            rank.publish(8).unwrap();
        }
    }

    #[test]
    fn generation_change_between_preflight_and_reserve_cancels_all() {
        let ranks: [_; 3] = std::array::from_fn(|_| KernargRetirement::new());
        let mut tickets = [None; 3];
        let reserved = Cell::new(0);
        let result = reserve_all(
            &mut tickets,
            |rank| ranks[rank].preflight(1, 4, || 0),
            |rank, ticket| {
                if rank == 1 {
                    ranks[rank].reserve(1, 4)?;
                    ranks[rank].publish(1)?;
                }
                ranks[rank].reserve_admitted(ticket)?;
                reserved.set(reserved.get() + 1);
                Ok(())
            },
            |rank| ranks[rank].cancel(),
        );
        assert_eq!(result, Err(Error::Stale));
        assert_eq!(reserved.get(), 1);
        for rank in &ranks {
            assert_eq!(rank.preflight(1, 4, || 0).unwrap_err(), Error::Pending);
        }
    }

    #[test]
    fn failure_after_actual_reservation_cancels_failed_and_unvisited_ranks() {
        let ranks: [_; 3] = std::array::from_fn(|_| KernargRetirement::new());
        let mut tickets = [None; 3];
        let result = reserve_all(
            &mut tickets,
            |rank| ranks[rank].preflight(1, 4, || 0),
            |rank, ticket| {
                ranks[rank].reserve_admitted(ticket)?;
                if rank == 1 {
                    Err(Error::Stale)
                } else {
                    Ok(())
                }
            },
            |rank| ranks[rank].cancel(),
        );
        assert_eq!(result, Err(Error::Stale));
        for rank in &ranks {
            assert_eq!(rank.complete(0, 0), Err(Error::Pending));
            assert_eq!(rank.reserve(1, 4), Err(Error::Pending));
        }
    }

    #[test]
    fn tickets_bind_queue_and_completion_generation() {
        let a = KernargRetirement::new();
        let b = KernargRetirement::new();
        a.reserve(4, 4).unwrap();
        a.publish(4).unwrap();
        let ticket = a.preflight(4, 4, || 0).unwrap();
        assert_eq!(b.reserve_admitted(ticket), Err(Error::Stale));
        a.complete(4, 0).unwrap();
        assert_eq!(a.reserve_admitted(ticket), Err(Error::Stale));
        let ticket = a.preflight(4, 4, || 0).unwrap();
        assert_eq!(a.reserve_admitted(ticket), Ok(4));
        assert_eq!(a.reserve_admitted(ticket), Err(Error::Pending));
        a.publish(8).unwrap();
        let result = a.preflight(4, 4, || {
            // Simulate another host submission while observing device completion.
            a.complete(8, 0).unwrap();
            a.reserve(1, 4).unwrap();
            a.publish(9).unwrap();
            0
        });
        assert_eq!(result.unwrap_err(), Error::Stale);
    }

    #[test]
    #[ignore = "CPU metadata only; excludes HSA and packet preparation"]
    fn collective_admission_cost() {
        use std::{hint::black_box, time::Instant};
        let mut samples = Vec::new();
        for _ in 0..9 {
            let ranks: [_; 8] = std::array::from_fn(|_| KernargRetirement::new());
            let mut tickets = [None; 8];
            let immediate = std::sync::atomic::AtomicBool::new(true);
            let start = Instant::now();
            for generation in 1..=100_000 {
                reserve_all(
                    black_box(&mut tickets),
                    |rank| black_box(&ranks[rank]).preflight(1, 4096, || black_box(0)),
                    |rank, ticket| black_box(&ranks[rank]).reserve_admitted(ticket).map(|_| ()),
                    |_| panic!("valid admission"),
                )
                .unwrap();
                for rank in &ranks {
                    black_box(super::retirement::ring_on_dispatch(true, || {
                        black_box(&immediate).load(std::sync::atomic::Ordering::Relaxed)
                    }));
                    black_box(super::retirement::ring_on_commit(
                        black_box(&immediate).load(std::sync::atomic::Ordering::Relaxed),
                    ));
                    black_box(rank).publish(black_box(generation)).unwrap();
                }
            }
            samples.push(start.elapsed().as_nanos() as f64 / 100_000.0);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "TP8 metadata admission median {:.2} ns/collective; samples {samples:?}",
            samples[4]
        );
        let mut baseline = Vec::new();
        for _ in 0..9 {
            let ranks: [_; 8] = std::array::from_fn(|_| KernargRetirement::new());
            let start = Instant::now();
            for generation in 1..=100_000 {
                for rank in &ranks {
                    match black_box(rank).reserve(1, 4096) {
                        Ok(_) => {}
                        Err(Error::Capacity) => {
                            let ticket = black_box(rank).completion_ticket().unwrap();
                            black_box(rank).complete(ticket, black_box(0)).unwrap();
                            black_box(rank).reserve(1, 4096).unwrap();
                        }
                        Err(error) => panic!("{error:?}"),
                    }
                }
                for rank in &ranks {
                    black_box(rank).publish(black_box(generation)).unwrap();
                }
            }
            baseline.push(start.elapsed().as_nanos() as f64 / 100_000.0);
        }
        baseline.sort_by(f64::total_cmp);
        eprintln!("TP8 prior metadata path median {:.2} ns/collective; admission delta {:.2} ns; samples {baseline:?}", baseline[4], samples[4] - baseline[4]);
    }
}
