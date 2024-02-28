use super::*;
use elliptic_curve::PrimeField;
use rand::{CryptoRng, Rng};
use serde::{de::Error as E, Deserializer, Serializer};
use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

/// A sparse polynomial over a prime field.
#[derive(Clone, PartialEq, Eq)]
pub struct SparsePolyPrimeField<F: PrimeField>(
    /// The coefficients and the powers of the polynomial
    pub BTreeMap<usize, F>,
);

unsafe impl<F: PrimeField> Send for SparsePolyPrimeField<F> {}

unsafe impl<F: PrimeField> Sync for SparsePolyPrimeField<F> {}

impl<F: PrimeField> Default for SparsePolyPrimeField<F> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<F: PrimeField> Debug for SparsePolyPrimeField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "SparsePolyPrimeField({:?})", self.0)
    }
}

impl<F: PrimeField> Display for SparsePolyPrimeField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let vals = self
            .0
            .iter()
            .map(|(power, c)| {
                let repr = c.to_repr();
                let len = repr.as_ref().len();
                let c = hex::encode(repr.as_ref());
                if *power == 0 {
                    let mut builder = "0x".to_string();
                    while builder.len() < len - 1 {
                        builder.push('0');
                    }
                    builder.push('1');
                    builder
                } else if *power == 1 {
                    format!("0x{}", c)
                } else {
                    format!("0x{}{}", c, to_super_script_digits(power + 1))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        write!(f, "{}", vals)
    }
}

impl<F: PrimeField> Add<&SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn add(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<F: PrimeField> Add<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn add(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        &self + rhs
    }
}

impl<F: PrimeField> Add<SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn add(self, rhs: SparsePolyPrimeField<F>) -> Self::Output {
        self + &rhs
    }
}

impl<F: PrimeField> Add for SparsePolyPrimeField<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<F: PrimeField> AddAssign<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn add_assign(&mut self, rhs: &SparsePolyPrimeField<F>) {
        for (exp, coeff) in &rhs.0 {
            match self.0.entry(*exp) {
                Entry::Occupied(e) => {
                    let new_coeff = e.remove() + coeff;
                    if new_coeff != F::ZERO {
                        self.0.insert(*exp, new_coeff);
                    }
                }
                Entry::Vacant(e) => {
                    if *coeff != F::ZERO {
                        e.insert(*coeff);
                    }
                }
            }
        }
    }
}

impl<F: PrimeField> AddAssign<SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn add_assign(&mut self, rhs: SparsePolyPrimeField<F>) {
        *self += &rhs;
    }
}

impl<F: PrimeField> Sub<&SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn sub(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        output -= rhs;
        output
    }
}

impl<F: PrimeField> Sub<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn sub(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        &self - rhs
    }
}

impl<F: PrimeField> Sub<SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn sub(self, rhs: SparsePolyPrimeField<F>) -> Self::Output {
        self - &rhs
    }
}

impl<F: PrimeField> Sub for SparsePolyPrimeField<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<F: PrimeField> SubAssign<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn sub_assign(&mut self, rhs: &SparsePolyPrimeField<F>) {
        for (exp, coeff) in &rhs.0 {
            match self.0.entry(*exp) {
                Entry::Occupied(e) => {
                    let new_coeff = e.remove() - coeff;
                    if new_coeff != F::ZERO {
                        self.0.insert(*exp, new_coeff);
                    }
                }
                Entry::Vacant(e) => {
                    if *coeff != F::ZERO {
                        e.insert(-*coeff);
                    }
                }
            }
        }
    }
}

impl<F: PrimeField> SubAssign<SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn sub_assign(&mut self, rhs: SparsePolyPrimeField<F>) {
        *self -= &rhs;
    }
}

impl<F: PrimeField> Neg for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn neg(self) -> Self::Output {
        let mut output = self.clone();
        for (_, c) in output.0.iter_mut() {
            *c = -(*c);
        }
        output
    }
}

impl<F: PrimeField> Neg for SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<F: PrimeField> Mul<&SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn mul(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<F: PrimeField> Mul<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn mul(self, rhs: &SparsePolyPrimeField<F>) -> Self::Output {
        &self * rhs
    }
}

impl<F: PrimeField> Mul<SparsePolyPrimeField<F>> for &SparsePolyPrimeField<F> {
    type Output = SparsePolyPrimeField<F>;

    fn mul(self, rhs: SparsePolyPrimeField<F>) -> Self::Output {
        self * &rhs
    }
}

impl<F: PrimeField> Mul for SparsePolyPrimeField<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<F: PrimeField> MulAssign<&SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn mul_assign(&mut self, rhs: &SparsePolyPrimeField<F>) {
        let mut result = SparsePolyPrimeField::default();

        for (exp1, coeff1) in &self.0 {
            for (exp2, coeff2) in &rhs.0 {
                let new_exp = exp1 + exp2;
                let mut new_coeff = *coeff1 * *coeff2;
                if new_coeff != F::ZERO {
                    match result.0.entry(new_exp) {
                        Entry::Occupied(e) => {
                            new_coeff += e.remove();
                            if new_coeff != F::ZERO {
                                result.0.insert(new_exp, new_coeff);
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert(new_coeff);
                        }
                    }
                }
            }
        }

        *self = result
    }
}

impl<F: PrimeField> MulAssign<SparsePolyPrimeField<F>> for SparsePolyPrimeField<F> {
    fn mul_assign(&mut self, rhs: SparsePolyPrimeField<F>) {
        *self *= &rhs;
    }
}

impl<F: PrimeField> Serialize for SparsePolyPrimeField<F> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            self.0
                .iter()
                .map(|(power, c)| (hex::encode(c.to_repr().as_ref()), power.to_string()))
                .collect::<Vec<_>>()
                .serialize(s)
        } else {
            let repr = F::Repr::default();
            let len = repr.as_ref().len();
            let mut rows = Vec::with_capacity(self.0.len());
            for (power, c) in &self.0 {
                let mut bytes = Vec::with_capacity(len + 8);
                let p = *power as u64;
                bytes.extend_from_slice(c.to_repr().as_ref());
                bytes.extend_from_slice(&p.to_be_bytes());
                rows.push(bytes);
            }
            rows.serialize(s)
        }
    }
}

impl<'de, F: PrimeField> Deserialize<'de> for SparsePolyPrimeField<F> {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if d.is_human_readable() {
            let v: Vec<(String, String)> = Vec::deserialize(d)?;
            let mut result = SparsePolyPrimeField::default();
            for (power, c) in &v {
                let repr_bytes = hex::decode(c).map_err(E::custom)?;
                let mut repr = F::Repr::default();
                repr.as_mut().copy_from_slice(&repr_bytes);
                let c = Option::<F>::from(F::from_repr(repr))
                    .ok_or(E::custom("invalid bytes for element"))?;
                let power = power.parse().map_err(E::custom)?;
                result.0.insert(power, c);
            }
            Ok(result)
        } else {
            let v: Vec<Vec<u8>> = Vec::deserialize(d)?;
            let mut result = SparsePolyPrimeField::default();
            for bytes in &v {
                let mut repr = F::Repr::default();
                let len = repr.as_ref().len();
                if bytes.len() < len + 8 {
                    return Err(E::custom("Invalid byte length"));
                }
                repr.as_mut().copy_from_slice(&bytes[..len]);
                let c = Option::<F>::from(F::from_repr(repr))
                    .ok_or(E::custom("Invalid field bytes"))?;
                let power = u64::from_be_bytes([
                    bytes[len - 8],
                    bytes[len - 7],
                    bytes[len - 6],
                    bytes[len - 5],
                    bytes[len - 4],
                    bytes[len - 3],
                    bytes[len - 2],
                    bytes[len - 1],
                ]) as usize;
                result.0.insert(power, c);
            }
            Ok(result)
        }
    }
}

impl<F: PrimeField> Polynomial<F> for SparsePolyPrimeField<F> {
    type X = F;
    const ZERO: Self = Self(BTreeMap::new());

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn one() -> Self {
        let mut map = BTreeMap::new();
        map.insert(0, F::ONE);
        Self(map)
    }

    fn degree(&self) -> usize {
        if let Some((power, _)) = self.0.last_key_value() {
            *power
        } else {
            0
        }
    }

    fn evaluate(&self, x: &Self::X) -> F {
        self.0.iter().fold(F::ZERO, move |acc, (power, c)| {
            acc + *c * x.pow([*power as u64])
        })
    }

    fn is_cyclotomic(&self) -> bool {
        if self.0.len() != 2 {
            return false;
        }
        if let Some(v) = self.0.get(&0) {
            if *v != -F::ONE {
                return false;
            }
        } else {
            return false;
        }
        if let Some((_, v)) = self.0.last_key_value() {
            if *v != F::ONE {
                return false;
            }
        } else {
            return false;
        }
        true
    }

    fn poly_mod(&self, m: &Self) -> (Self, Self) {
        // Ensure divisor is not zero
        assert!(!m.0.is_empty());

        let self_degree = self.degree();
        let m_degree = m.degree();
        if self_degree < m_degree {
            return (Self::ZERO, self.clone());
        }

        let mut quotient = SparsePolyPrimeField(BTreeMap::new());
        let mut remainder = self.clone();

        // Loop until the remainder's degree is less than the divisor's degree
        let lead_term_div = m.0.last_key_value().expect("should be at least one entry");
        let largest_coeff_inv = lead_term_div
            .1
            .invert()
            .expect("lead term should not be zero");
        while !remainder.0.is_empty() && remainder.degree() >= m_degree {
            // Calculate the leading term of the remainder and divisor
            let lead_term_rem = remainder
                .0
                .last_key_value()
                .expect("remainder should have at least one entry");

            // Calculate the exponent and coefficient for the division
            let exp_diff = lead_term_rem.0 - lead_term_div.0;
            let coeff_div = *lead_term_rem.1 * largest_coeff_inv;

            if coeff_div == F::ZERO {
                continue;
            }

            // Add the term to the quotient
            quotient.0.insert(exp_diff, coeff_div);

            // Subtract the term (divisor * coeff_div * x^exp_diff) from the remainder
            let term_to_subtract =
                m.0.iter()
                    .map(|(exp, coeff)| (*exp + exp_diff, *coeff * coeff_div))
                    .collect::<BTreeMap<_, _>>();
            remainder -= SparsePolyPrimeField(term_to_subtract);
        }

        (quotient, remainder)
    }
}

impl<F: PrimeField> SparsePolyPrimeField<F> {
    /// Generate a random sparse polynomial where the length is defined by the `num_terms`
    /// and the powers are randomly less than `max_power`
    pub fn random(num_terms: usize, max_power: usize, mut rng: impl RngCore + CryptoRng) -> Self {
        let mut coeffs = BTreeMap::new();
        while coeffs.len() < num_terms {
            let power = rng.gen::<usize>() % max_power;
            let s = F::random(&mut rng);
            coeffs.entry(power).and_modify(|c| *c += s).or_insert(s);
        }
        Self(coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    fn add() {
        let mut rng = ChaChaRng::from_seed([4u8; 32]);
        let a = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        let b = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        assert!(a.degree() < 100);
        assert!(b.degree() < 100);
        let c = &a + &b;
        assert_eq!(c.0.len(), 7);
    }

    #[test]
    fn sub() {
        let mut rng = ChaChaRng::from_seed([4u8; 32]);
        let a = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        let b = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        assert!(a.degree() < 100);
        assert!(b.degree() < 100);
        let c = &a - &b;
        assert_eq!(c.0.len(), 7);
    }

    #[test]
    fn mul() {
        let mut rng = ChaChaRng::from_seed([8u8; 32]);
        let a = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        let b = SparsePolyPrimeField::<k256::Scalar>::random(4, 100, &mut rng);
        let c = &a * &b;
        println!("{}", c);
    }

    #[test]
    fn poly_mod() {
        // x^4 - 2x^2 - 4
        let mut dividend = SparsePolyPrimeField::default();
        dividend.0.insert(3, k256::Scalar::ONE);
        dividend.0.insert(2, -k256::Scalar::from(2u32));
        dividend.0.insert(0, -k256::Scalar::from(4u32));

        // x - 3
        let mut divisor = SparsePolyPrimeField::default();
        divisor.0.insert(1, k256::Scalar::ONE);
        divisor.0.insert(0, -k256::Scalar::from(3u32));

        let (quotient, remainder) = dividend.poly_mod(&divisor);

        assert_eq!(quotient.0.len(), 3);
        assert_eq!(quotient.0.get(&2), Some(k256::Scalar::ONE).as_ref());
        assert_eq!(quotient.0.get(&1), Some(k256::Scalar::ONE).as_ref());
        assert_eq!(quotient.0.get(&0), Some(k256::Scalar::from(3u32)).as_ref());
        assert_eq!(remainder.0.len(), 1);
        assert_eq!(remainder.0.get(&0), Some(k256::Scalar::from(5u32)).as_ref());

        let mut res = quotient * divisor;
        res += remainder;
        assert_eq!(res, dividend);

        let mut rng = ChaChaRng::from_seed([9u8; 32]);

        let a = SparsePolyPrimeField::<k256::Scalar>::random(4, 20, &mut rng);
        let b = SparsePolyPrimeField::<k256::Scalar>::random(2, 10, &mut rng);

        let (div, rem) = a.poly_mod(&b);
        let div_b = &div * &b;
        let div_b_pr = &div_b + &rem;
        assert_eq!(a, div_b_pr);

        let (div, rem) = b.poly_mod(&a);
        assert_eq!(div, SparsePolyPrimeField::default());
        assert_eq!(rem, b);
    }

    #[test]
    fn poly_mod_cyclotomic() {
        let mut rng = ChaChaRng::from_seed([9u8; 32]);
        let a = SparsePolyPrimeField::<k256::Scalar>::random(10, 100, &mut rng);
        let mut b = SparsePolyPrimeField::default();
        b.0.insert(a.degree() / 2, k256::Scalar::ONE);
        b.0.insert(0, -k256::Scalar::ONE);

        let (div, rem) = a.poly_mod(&b);
        let div_b = &div * &b;
        let div_b_pr = &div_b + &rem;
        assert_eq!(a, div_b_pr);
    }
}
