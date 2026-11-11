//! Per-frame cover of the live rows of a staged program family (depth / RVQ stages) by the row
//! capacities its packet declares. The caller packs the live rows (row `i` of the plan is the
//! `i`-th live stream, wherever its slot sits), so a launch of capacity `c` at `start` computes
//! rows `start..start + c`, and the plan is the cheapest set of back-to-back windows holding the
//! live count, by each program's measured launch cost. Nothing here caps rows or holds a plan:
//! every frame is planned from that frame's live count.
//!
//! Costs are microseconds, measured once at load and then averaged over the frames a program runs
//! alone. Until every candidate has a cost the plan is the single smallest covering program.

/// One launch: `program` (index into the family) over rows `start..start + capacity`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Window {
    pub program: u16,
    pub start: u32,
}

/// Which programs a plan may use (`PLOW_TTS_DEPTH_PLAN`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Cheapest cover over every declared capacity.
    Auto,
    /// No packing: the single smallest program covering the highest live slot.
    Cover,
    /// Cheapest cover over these capacities only (debugging).
    Only(Vec<u32>),
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "" | "auto" => Ok(Self::Auto),
            "cover" => Ok(Self::Cover),
            list => list
                .split(',')
                .map(|v| {
                    v.trim()
                        .parse::<u32>()
                        .map_err(|_| format!("depth plan: {v:?}"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Self::Only),
        }
    }
}

const UNMEASURED: f32 = 0.0;

pub struct RowPlan {
    caps: Vec<u32>,
    allowed: Vec<bool>,
    cost: Vec<f32>,
    dynamic: bool,
    best: Vec<f32>,
    pick: Vec<u16>,
    plan: Vec<Window>,
}

impl RowPlan {
    /// `caps[i]` is program `i`'s row capacity; at most `rows` rows are ever live.
    pub fn new(caps: &[u32], rows: u32, mode: &Mode) -> Self {
        let allowed = caps
            .iter()
            .map(|c| match mode {
                Mode::Only(only) => only.contains(c),
                _ => true,
            })
            .collect();
        let n = rows as usize;
        Self {
            caps: caps.to_vec(),
            allowed,
            cost: vec![UNMEASURED; caps.len()],
            dynamic: *mode != Mode::Cover,
            best: vec![0.0; n + 1],
            pick: vec![0; n + 1],
            plan: Vec::with_capacity(n),
        }
    }

    pub fn dynamic(&self) -> bool {
        self.dynamic
    }

    pub fn capacity(&self, program: u16) -> u32 {
        self.caps[usize::from(program)]
    }

    /// Programs a plan may pick, for the load-time measurement.
    pub fn candidates(&self) -> impl Iterator<Item = u16> + '_ {
        (0..self.caps.len())
            .filter(|&p| self.allowed[p])
            .map(|p| p as u16)
    }

    pub fn cost_us(&self, program: u16) -> f32 {
        self.cost[usize::from(program)]
    }

    /// A launch of `program` alone took `us`.
    pub fn observe(&mut self, program: u16, us: f32) {
        if !(us.is_finite() && us > 0.0) {
            return;
        }
        let c = &mut self.cost[usize::from(program)];
        *c = if *c == UNMEASURED {
            us
        } else {
            *c + 0.2 * (us - *c)
        };
    }

    /// The last plan.
    pub fn windows(&self) -> &[Window] {
        &self.plan
    }

    /// The windows for `live` packed rows, widest first; empty when no program covers them.
    pub fn plan(&mut self, live: usize) -> &[Window] {
        self.plan.clear();
        if live == 0 || !(self.dynamic && self.cover(live)) {
            self.plan.clear();
            self.single(live);
        }
        &self.plan
    }

    /// The single smallest program of at least `width` rows.
    pub fn single(&mut self, width: usize) -> &[Window] {
        self.plan.clear();
        if let Some(p) = (0..self.caps.len())
            .filter(|&p| width > 0 && self.caps[p] as usize >= width)
            .min_by_key(|&p| self.caps[p])
        {
            self.plan.push(Window {
                program: p as u16,
                start: 0,
            });
        }
        &self.plan
    }

    /// Cheapest multiset of measured candidates holding `live` rows.
    fn cover(&mut self, live: usize) -> bool {
        if live >= self.best.len()
            || (0..self.caps.len()).any(|p| self.allowed[p] && self.cost[p] == UNMEASURED)
        {
            return false;
        }
        self.best[0] = 0.0;
        for j in 1..=live {
            let (mut best, mut best_cap) = (f32::INFINITY, 0);
            for p in 0..self.caps.len() {
                let cap = self.caps[p];
                if !self.allowed[p] || cap == 0 {
                    continue;
                }
                let total = self.cost[p] + self.best[j.saturating_sub(cap as usize)];
                // Equal covers: the wider program, so a stream's capacity does not flip between
                // equivalent plans.
                if total < best * (1.0 - 1e-6) || (total <= best * (1.0 + 1e-6) && cap > best_cap) {
                    (best, best_cap) = (total, cap);
                    self.pick[j] = p as u16;
                }
            }
            self.best[j] = best;
        }
        if !self.best[live].is_finite() {
            return false;
        }
        let mut left = live;
        while left > 0 {
            let program = self.pick[left];
            self.plan.push(Window { program, start: 0 });
            left = left.saturating_sub(self.caps[usize::from(program)] as usize);
        }
        let caps = &self.caps;
        self.plan
            .sort_unstable_by_key(|w| std::cmp::Reverse(caps[usize::from(w.program)]));
        let mut start = 0;
        for w in &mut self.plan {
            w.start = start;
            start += caps[usize::from(w.program)];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(caps: &[u32], rows: u32, mode: Mode, cost: impl Fn(u32) -> f32) -> RowPlan {
        let mut p = RowPlan::new(caps, rows, &mode);
        for (i, &c) in caps.iter().enumerate() {
            p.observe(i as u16, cost(c));
        }
        p
    }

    fn caps_of(p: &mut RowPlan, live: usize) -> Vec<(u32, u32)> {
        let w = p.plan(live).to_vec();
        w.iter().map(|w| (p.capacity(w.program), w.start)).collect()
    }

    const AFFINE: fn(u32) -> f32 = |c| 12.0 + 1.7 * c as f32;

    #[test]
    fn unmeasured_and_cover_take_the_smallest_covering_program() {
        let caps = [1, 2, 4, 8, 16, 32];
        let mut p = RowPlan::new(&caps, 32, &Mode::Auto);
        assert_eq!(caps_of(&mut p, 20), [(32, 0)]);
        let mut p = measured(&caps, 32, Mode::Cover, AFFINE);
        assert_eq!(caps_of(&mut p, 20), [(32, 0)]);
        assert!(p.plan(40).is_empty());
        assert!(p.plan(0).is_empty());
    }

    #[test]
    fn splits_only_where_the_measured_costs_say_so() {
        let caps = [1, 2, 4, 8, 16, 32];
        let mut p = measured(&caps, 32, Mode::Auto, AFFINE);
        assert_eq!(caps_of(&mut p, 20), [(16, 0), (4, 16)]);
        assert_eq!(caps_of(&mut p, 28), [(32, 0)]);
        assert_eq!(caps_of(&mut p, 12), [(16, 0)]);
        let mut p = measured(&[1, 2, 4, 8, 16, 20, 24, 32], 32, Mode::Auto, AFFINE);
        assert_eq!(caps_of(&mut p, 20), [(20, 0)]);
        assert_eq!(caps_of(&mut p, 21), [(24, 0)]);
    }

    #[test]
    fn restricted_plans_and_the_widest_rung() {
        let caps = [1, 2, 4, 8, 16, 32];
        let mut p = measured(&caps, 32, Mode::Only(vec![4]), AFFINE);
        assert_eq!(caps_of(&mut p, 16), [(4, 0), (4, 4), (4, 8), (4, 12)]);
        assert_eq!(caps_of(&mut p, 3), [(4, 0)]);
        let mut p = measured(&caps, 32, Mode::Only(vec![16, 4]), AFFINE);
        assert_eq!(caps_of(&mut p, 32), [(16, 0), (16, 16)]);
        assert_eq!(caps_of(&mut p, 31), [(16, 0), (16, 16)]);
    }
}
