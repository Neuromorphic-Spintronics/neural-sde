import Mathlib

noncomputable section

namespace ScientificResearchSystems

open Real

def s17 : ℝ := Real.sqrt 17

def cy : ℝ := (85 + 19 * s17) / 34
def cz : ℝ := (119 + 9 * s17) / 34
def cw : ℝ := (51 + 23 * s17) / 34

lemma s17_nonneg : 0 ≤ s17 := by
  simp [s17]

lemma s17_sq : s17 ^ 2 = 17 := by
  norm_num [s17]

lemma s17_lower : (41 : ℝ) / 10 < s17 := by
  have hs0 := s17_nonneg
  have hs2 := s17_sq
  by_contra h
  have hle : s17 ≤ (41 : ℝ) / 10 := le_of_not_gt h
  nlinarith [sq_nonneg (s17 - (41 : ℝ) / 10)]

lemma s17_upper : s17 < (33 : ℝ) / 8 := by
  have hs0 := s17_nonneg
  have hs2 := s17_sq
  by_contra h
  have hge : (33 : ℝ) / 8 ≤ s17 := le_of_not_gt h
  nlinarith [sq_nonneg (s17 - (33 : ℝ) / 8)]

lemma cy_pos : 0 < cy := by
  dsimp [cy]
  nlinarith [s17_nonneg]

lemma cz_sub_four_pos : 0 < cz - 4 := by
  dsimp [cz]
  nlinarith [s17_lower]

lemma cw_sub_one_pos : 0 < cw - 1 := by
  dsimp [cw]
  nlinarith [s17_nonneg]

lemma cw_sub_four_pos : 0 < cw - 4 := by
  dsimp [cw]
  nlinarith [s17_lower]

lemma delta0_pos :
    0 < 4 * cy * cz - (cy + cz - 16) ^ 2 := by
  have hs2 := s17_sq
  have hslo := s17_lower
  dsimp [cy, cz]
  nlinarith

def p2 (r : ℝ) : ℝ :=
  ((111 - 23 * s17) / 17) * r + (145 + 23 * s17) / 17

def p1 (r : ℝ) : ℝ :=
  ((267 + 77 * s17) / 34) * r ^ 2
    - ((766 + 82 * s17) / 17) * r
    + (241 + 87 * s17) / 34

def p0 (r : ℝ) : ℝ :=
  (cy * r + cz - 4) *
    (cy * r ^ 2 + (cy + cz - 16) * r + cz)

def B (r : ℝ) : ℝ :=
  r ^ 2 + ((2764 : ℝ) / 9 - 78 * s17) * r
    + (749 : ℝ) / 9 - 20 * s17

lemma p2_pos {r : ℝ} (hr : 0 ≤ r) : 0 < p2 r := by
  have hsup := s17_upper
  have hs0 := s17_nonneg
  have hslope : 0 < (111 : ℝ) - 23 * s17 := by nlinarith
  have hinter : 0 < (145 : ℝ) + 23 * s17 := by nlinarith
  dsimp [p2]
  nlinarith

lemma p0_pos {r : ℝ} (hr : 0 ≤ r) : 0 < p0 r := by
  have hcy := cy_pos
  have hcz4 := cz_sub_four_pos
  have hdelta := delta0_pos
  have hfac1 : 0 < cy * r + cz - 4 := by
    have : 0 ≤ cy * r := mul_nonneg hcy.le hr
    nlinarith
  have hsquare : 0 ≤ (2 * cy * r + (cy + cz - 16)) ^ 2 := sq_nonneg _
  have hid :
      4 * cy * (cy * r ^ 2 + (cy + cz - 16) * r + cz)
        = (2 * cy * r + (cy + cz - 16)) ^ 2
          + (4 * cy * cz - (cy + cz - 16) ^ 2) := by
    ring
  have hprod : 0 < 4 * cy * (cy * r ^ 2 + (cy + cz - 16) * r + cz) := by
    rw [hid]
    nlinarith
  have hfac2 : 0 < cy * r ^ 2 + (cy + cz - 16) * r + cz := by
    nlinarith
  exact mul_pos hfac1 hfac2

lemma p1_quarter_pos : 0 < p1 ((1 : ℝ) / 4) := by
  have hslo := s17_lower
  dsimp [p1]
  nlinarith

lemma p1_four_pos : 0 < p1 4 := by
  have hslo := s17_lower
  dsimp [p1]
  nlinarith

lemma p1_left_factor_neg :
    ((267 + 77 * s17) / 34) * ((1 : ℝ) / 2)
      - (766 + 82 * s17) / 17 < 0 := by
  have hsup := s17_upper
  nlinarith

lemma p1_right_factor_pos :
    ((267 + 77 * s17) / 34) * 8
      - (766 + 82 * s17) / 17 > 0 := by
  have hslo := s17_lower
  nlinarith

lemma p1_neg_interval {r : ℝ} (hneg : p1 r < 0) :
    (1 : ℝ) / 4 < r ∧ r < 4 := by
  constructor
  · by_contra h
    have hr : r ≤ (1 : ℝ) / 4 := le_of_not_gt h
    let a : ℝ := (267 + 77 * s17) / 34
    let b : ℝ := - (766 + 82 * s17) / 17
    have ha : 0 < a := by
      dsimp [a]
      nlinarith [s17_nonneg]
    have hsec : a * (r + (1 : ℝ) / 4) + b ≤ 0 := by
      have hsum : r + (1 : ℝ) / 4 ≤ (1 : ℝ) / 2 := by nlinarith
      have hmul := mul_le_mul_of_nonneg_left hsum ha.le
      dsimp [a, b] at hmul ⊢
      have hf := p1_left_factor_neg
      nlinarith
    have hprod : 0 ≤ (r - (1 : ℝ) / 4) * (a * (r + (1 : ℝ) / 4) + b) :=
      mul_nonneg_of_nonpos_of_nonpos (by nlinarith) hsec
    have hid :
        p1 r - p1 ((1 : ℝ) / 4)
          = (r - (1 : ℝ) / 4) * (a * (r + (1 : ℝ) / 4) + b) := by
      dsimp [p1, a, b]
      ring
    rw [← hid] at hprod
    nlinarith [p1_quarter_pos]
  · by_contra h
    have hr : 4 ≤ r := le_of_not_gt h
    let a : ℝ := (267 + 77 * s17) / 34
    let b : ℝ := - (766 + 82 * s17) / 17
    have ha : 0 < a := by
      dsimp [a]
      nlinarith [s17_nonneg]
    have hsec : 0 ≤ a * (r + 4) + b := by
      have hsum : 8 ≤ r + 4 := by nlinarith
      have hmul := mul_le_mul_of_nonneg_left hsum ha.le
      dsimp [a, b] at hmul ⊢
      have hf := p1_right_factor_pos
      nlinarith
    have hprod : 0 ≤ (r - 4) * (a * (r + 4) + b) :=
      mul_nonneg (by nlinarith) hsec
    have hid :
        p1 r - p1 4 = (r - 4) * (a * (r + 4) + b) := by
      dsimp [p1, a, b]
      ring
    rw [← hid] at hprod
    nlinarith [p1_four_pos]

lemma B_quarter_neg : B ((1 : ℝ) / 4) < 0 := by
  have hslo := s17_lower
  dsimp [B]
  nlinarith

lemma B_four_neg : B 4 < 0 := by
  have hslo := s17_lower
  dsimp [B]
  nlinarith

lemma B_neg_of_interval {r : ℝ} (hlo : (1 : ℝ) / 4 < r) (hhi : r < 4) :
    B r < 0 := by
  have hright : 0 < r - (1 : ℝ) / 4 := by nlinarith
  have hleft : 0 < 4 - r := by nlinarith
  have hprod : (r - (1 : ℝ) / 4) * (r - 4) < 0 :=
    mul_neg_of_pos_of_neg hright (by nlinarith)
  have h1 : (4 - r) * B ((1 : ℝ) / 4) < 0 :=
    mul_neg_of_pos_of_neg hleft B_quarter_neg
  have h2 : (r - (1 : ℝ) / 4) * B 4 < 0 :=
    mul_neg_of_pos_of_neg hright B_four_neg
  have hid :
      ((15 : ℝ) / 4) * B r
        = (4 - r) * B ((1 : ℝ) / 4)
          + (r - (1 : ℝ) / 4) * B 4
          + ((15 : ℝ) / 4) * (r - (1 : ℝ) / 4) * (r - 4) := by
    dsimp [B]
    ring
  rw [hid]
  nlinarith

lemma discriminant_factor (r : ℝ) :
    4 * p2 r * p0 r - (p1 r) ^ 2
      = - ((3537 + 855 * s17) / 34) * (r - 1) ^ 2 * B r := by
  have hs2 := s17_sq
  dsimp [p2, p1, p0, B, cy, cz]
  ring_nf
  nlinarith

lemma reduced_nonneg {r u : ℝ} (hr : 0 ≤ r) (hu : 0 ≤ u) :
    0 ≤ p2 r * u ^ 2 + p1 r * u + p0 r := by
  have hp2 := p2_pos hr
  have hp0 := p0_pos hr
  by_cases h1 : 0 ≤ p1 r
  · have hu2 : 0 ≤ u ^ 2 := sq_nonneg u
    have ht1 : 0 ≤ p2 r * u ^ 2 := mul_nonneg hp2.le hu2
    have ht2 : 0 ≤ p1 r * u := mul_nonneg h1 hu
    nlinarith
  · have h1neg : p1 r < 0 := lt_of_not_ge h1
    obtain ⟨hlo, hhi⟩ := p1_neg_interval h1neg
    have hB := B_neg_of_interval hlo hhi
    have hcoef : 0 < (3537 + 855 * s17) / 34 := by
      nlinarith [s17_nonneg]
    have hdisc : 0 ≤ 4 * p2 r * p0 r - (p1 r) ^ 2 := by
      rw [discriminant_factor r]
      have hsquare : 0 ≤ (r - 1) ^ 2 := sq_nonneg _
      have htmp : 0 ≤ - ((3537 + 855 * s17) / 34) * B r := by
        exact (mul_pos (by nlinarith) hB).le
      nlinarith [mul_nonneg htmp hsquare]
    have hsquare : 0 ≤ (2 * p2 r * u + p1 r) ^ 2 := sq_nonneg _
    have hid :
        4 * p2 r * (p2 r * u ^ 2 + p1 r * u + p0 r)
          = (2 * p2 r * u + p1 r) ^ 2
            + (4 * p2 r * p0 r - (p1 r) ^ 2) := by ring
    have hprod : 0 ≤ 4 * p2 r * (p2 r * u ^ 2 + p1 r * u + p0 r) := by
      rw [hid]
      nlinarith
    nlinarith

lemma reduced_identity (r u : ℝ) :
    (cy * r + cz - 4 + (cw - 1) * u) *
        (cy * r ^ 2 + cz + (cy + cz - 16) * r
          + (cw - 4) * r * u + cw * u)
      - 64 * r * u
      = p2 r * u ^ 2 + p1 r * u + p0 r := by
  have hs2 := s17_sq
  dsimp [p2, p1, p0, cy, cz, cw]
  ring_nf
  nlinarith

lemma cubic_nonneg (Y Z W : ℝ) (hY : 0 ≤ Y) (hZ : 0 ≤ Z) (hW : 0 ≤ W) :
    0 ≤
      (cy * Y + (cz - 4) * Z + (cw - 1) * W) *
        (cy * Y ^ 2 + cz * Z ^ 2 + (cy + cz - 16) * Y * Z
          + (cw - 4) * Y * W + cw * Z * W)
      - 64 * Y * Z * W := by
  by_cases hZ0 : Z = 0
  · subst Z
    have hA : 0 ≤ cy * Y + (cw - 1) * W := by
      exact add_nonneg (mul_nonneg cy_pos.le hY) (mul_nonneg cw_sub_one_pos.le hW)
    have hD : 0 ≤ cy * Y ^ 2 + (cw - 4) * Y * W := by
      exact add_nonneg (mul_nonneg cy_pos.le (sq_nonneg Y))
        (mul_nonneg (mul_nonneg cw_sub_four_pos.le hY) hW)
    norm_num
    exact mul_nonneg hA hD
  · have hZpos : 0 < Z := lt_of_le_of_ne hZ (Ne.symm hZ0)
    let r : ℝ := Y / Z
    let u : ℝ := W / Z
    have hr : 0 ≤ r := div_nonneg hY hZ
    have hu : 0 ≤ u := div_nonneg hW hZ
    have hred := reduced_nonneg hr hu
    have hidred := reduced_identity r u
    have hscale :
        (cy * Y + (cz - 4) * Z + (cw - 1) * W) *
            (cy * Y ^ 2 + cz * Z ^ 2 + (cy + cz - 16) * Y * Z
              + (cw - 4) * Y * W + cw * Z * W)
          - 64 * Y * Z * W
        = Z ^ 3 *
            ((cy * r + cz - 4 + (cw - 1) * u) *
                (cy * r ^ 2 + cz + (cy + cz - 16) * r
                  + (cw - 4) * r * u + cw * u)
              - 64 * r * u) := by
      dsimp [r, u]
      field_simp [hZ0]
      ring
    rw [hscale, hidred]
    exact mul_nonneg (by positivity) hred

theorem sharp_quartic (x y z w : ℝ) :
    (x * w + 4 * y * z) ^ 2 + 4 * (x * z + y * w) ^ 2
      ≤ (x ^ 2 + y ^ 2 + z ^ 2) *
          (cy * y ^ 2 + cz * z ^ 2 + cw * w ^ 2) := by
  let Y : ℝ := y ^ 2
  let Z : ℝ := z ^ 2
  let W : ℝ := w ^ 2
  let A : ℝ := cy * Y + (cz - 4) * Z + (cw - 1) * W
  let D : ℝ := cy * Y ^ 2 + cz * Z ^ 2 + (cy + cz - 16) * Y * Z
    + (cw - 4) * Y * W + cw * Z * W
  let Q : ℝ :=
    (x ^ 2 + y ^ 2 + z ^ 2) *
        (cy * y ^ 2 + cz * z ^ 2 + cw * w ^ 2)
      - ((x * w + 4 * y * z) ^ 2 + 4 * (x * z + y * w) ^ 2)
  have hY : 0 ≤ Y := by dsimp [Y]; positivity
  have hZ : 0 ≤ Z := by dsimp [Z]; positivity
  have hW : 0 ≤ W := by dsimp [W]; positivity
  have hA0 : 0 ≤ A := by
    dsimp [A]
    exact add_nonneg (add_nonneg (mul_nonneg cy_pos.le hY)
      (mul_nonneg cz_sub_four_pos.le hZ)) (mul_nonneg cw_sub_one_pos.le hW)
  have hAD : 0 ≤ A * D - 64 * Y * Z * W := by
    exact cubic_nonneg Y Z W hY hZ hW
  have hQexp : Q = A * x ^ 2 - 16 * y * z * w * x + D := by
    dsimp [Q, A, D, Y, Z, W]
    ring
  by_cases hA : A = 0
  · have hy : y = 0 := by
      by_contra hy
      have ht : 0 < cy * Y := mul_pos cy_pos (by dsimp [Y]; positivity)
      have hzterm : 0 ≤ (cz - 4) * Z := mul_nonneg cz_sub_four_pos.le hZ
      have hwterm : 0 ≤ (cw - 1) * W := mul_nonneg cw_sub_one_pos.le hW
      dsimp [A] at hA
      nlinarith
    have hz : z = 0 := by
      by_contra hz
      have ht : 0 < (cz - 4) * Z := mul_pos cz_sub_four_pos (by dsimp [Z]; positivity)
      have hyterm : 0 ≤ cy * Y := mul_nonneg cy_pos.le hY
      have hwterm : 0 ≤ (cw - 1) * W := mul_nonneg cw_sub_one_pos.le hW
      dsimp [A] at hA
      nlinarith
    have hw : w = 0 := by
      by_contra hw
      have ht : 0 < (cw - 1) * W := mul_pos cw_sub_one_pos (by dsimp [W]; positivity)
      have hyterm : 0 ≤ cy * Y := mul_nonneg cy_pos.le hY
      have hzterm : 0 ≤ (cz - 4) * Z := mul_nonneg cz_sub_four_pos.le hZ
      dsimp [A] at hA
      nlinarith
    subst y
    subst z
    subst w
    norm_num
  · have hApos : 0 < A := lt_of_le_of_ne hA0 (Ne.symm hA)
    have hid :
        A * Q = (A * x - 8 * y * z * w) ^ 2 + (A * D - 64 * Y * Z * W) := by
      rw [hQexp]
      dsimp [Y, Z, W]
      ring
    have hAQ : 0 ≤ A * Q := by
      rw [hid]
      exact add_nonneg (sq_nonneg _) hAD
    have hQ : 0 ≤ Q := nonneg_of_mul_nonneg_left hAQ hApos
    dsimp [Q] at hQ
    linarith

def kappa : ℝ := (3 * s17 - 5) / 8

lemma kappa_pos : 0 < kappa := by
  dsimp [kappa]
  nlinarith [s17_lower]

lemma kappa_lt_one : kappa < 1 := by
  dsimp [kappa]
  nlinarith [s17_upper]

lemma kappa_equation : 4 * kappa ^ 2 + 5 * kappa - 8 = 0 := by
  have hs2 := s17_sq
  dsimp [kappa]
  nlinarith

def familyInfo (u : ℝ) : ℝ :=
  (5 * u ^ 2 + 16 * u + 20) / (3 * (u ^ 2 + 2))

lemma familyInfo_kappa : familyInfo kappa = (5 + s17) / 2 := by
  have hk := kappa_equation
  have hden : kappa ^ 2 + 2 ≠ 0 := by positivity
  dsimp [familyInfo]
  field_simp [hden]
  have hs2 := s17_sq
  dsimp [kappa] at *
  nlinarith

lemma sharp_gap_identity :
    5 - (5 + s17) / 2 = (5 - s17) / 2 := by
  ring

end ScientificResearchSystems
