/*
    Copyright LIT Protocol . All Rights Reserved.
    SPDX-License-Identifier: FSL-1.1
*/
use super::*;
use elliptic_curve::PrimeField;
use rayon::prelude::*;
use serde::{
    de::{Error as E, Visitor},
    Deserializer, Serializer,
};
use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    marker::PhantomData,
};

/// A polynomial over a prime field
#[derive(Clone, PartialEq, Eq)]
pub struct DensePrimeField<F: PrimeField>(
    /// The coefficients of the polynomial
    pub Vec<F>,
);

unsafe impl<F: PrimeField> Send for DensePrimeField<F> {}

unsafe impl<F: PrimeField> Sync for DensePrimeField<F> {}

impl<F: PrimeField> Default for DensePrimeField<F> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<F: PrimeField> Debug for DensePrimeField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "PolyPrimeField({:?})", self.0)
    }
}

impl<F: PrimeField> Display for DensePrimeField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let vals = self
            .0
            .iter()
            .enumerate()
            .map(|(power, c)| {
                let c = hex::encode(c.to_repr().as_ref());
                if power == 0 {
                    c
                } else {
                    format!("{}{}", c, to_super_script_digits(power + 1))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        write!(f, "{}", vals)
    }
}

impl<F: PrimeField> Add<&DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn add(self, rhs: &DensePrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        add_poly(&mut output.0, &rhs.0);
        output
    }
}

impl<F: PrimeField> Add<&DensePrimeField<F>> for DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn add(self, rhs: &DensePrimeField<F>) -> Self::Output {
        &self + rhs
    }
}

impl<F: PrimeField> Add<DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn add(self, rhs: DensePrimeField<F>) -> Self::Output {
        self + &rhs
    }
}

impl<F: PrimeField> Add for DensePrimeField<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<F: PrimeField> AddAssign<&DensePrimeField<F>> for DensePrimeField<F> {
    fn add_assign(&mut self, rhs: &DensePrimeField<F>) {
        add_poly(&mut self.0, &rhs.0);
    }
}

impl<F: PrimeField> AddAssign<DensePrimeField<F>> for DensePrimeField<F> {
    fn add_assign(&mut self, rhs: DensePrimeField<F>) {
        add_poly(&mut self.0, &rhs.0);
    }
}

impl<F: PrimeField> Sub<&DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn sub(self, rhs: &DensePrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        sub_poly(&mut output.0, &rhs.0);
        output
    }
}

impl<F: PrimeField> Sub<&DensePrimeField<F>> for DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn sub(self, rhs: &DensePrimeField<F>) -> Self::Output {
        &self - rhs
    }
}

impl<F: PrimeField> Sub<DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn sub(self, rhs: DensePrimeField<F>) -> Self::Output {
        self - &rhs
    }
}

impl<F: PrimeField> Sub for DensePrimeField<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<F: PrimeField> SubAssign<&DensePrimeField<F>> for DensePrimeField<F> {
    fn sub_assign(&mut self, rhs: &DensePrimeField<F>) {
        sub_poly(&mut self.0, &rhs.0);
    }
}

impl<F: PrimeField> SubAssign<DensePrimeField<F>> for DensePrimeField<F> {
    fn sub_assign(&mut self, rhs: DensePrimeField<F>) {
        sub_poly(&mut self.0, &rhs.0);
    }
}

impl<F: PrimeField> Neg for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn neg(self) -> Self::Output {
        let mut output = self.clone();
        output.0.par_iter_mut().for_each(|c| *c = -*c);
        output
    }
}

impl<F: PrimeField> Neg for DensePrimeField<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<F: PrimeField> Mul<&DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn mul(self, rhs: &DensePrimeField<F>) -> Self::Output {
        let mut output = self.clone();
        mul_poly(&mut output.0, &rhs.0, F::ZERO);
        output
    }
}

impl<F: PrimeField> Mul<&DensePrimeField<F>> for DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn mul(self, rhs: &DensePrimeField<F>) -> Self::Output {
        &self * rhs
    }
}

impl<F: PrimeField> Mul<DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn mul(self, rhs: DensePrimeField<F>) -> Self::Output {
        self * &rhs
    }
}

impl<F: PrimeField> Mul for DensePrimeField<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<F: PrimeField> Mul<&F> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn mul(self, rhs: &F) -> Self::Output {
        self * *rhs
    }
}

impl<F: PrimeField> Mul<F> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn mul(self, rhs: F) -> Self::Output {
        let output = self.0.par_iter().map(|s| *s * rhs).collect();
        DensePrimeField(output)
    }
}

impl<F: PrimeField> Mul<&F> for DensePrimeField<F> {
    type Output = Self;

    fn mul(self, rhs: &F) -> Self::Output {
        &self * *rhs
    }
}

impl<F: PrimeField> Mul<F> for DensePrimeField<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        &self * rhs
    }
}

impl<F: PrimeField> MulAssign<&DensePrimeField<F>> for DensePrimeField<F> {
    fn mul_assign(&mut self, rhs: &DensePrimeField<F>) {
        mul_poly(&mut self.0, &rhs.0, F::ZERO);
    }
}

impl<F: PrimeField> MulAssign<DensePrimeField<F>> for DensePrimeField<F> {
    fn mul_assign(&mut self, rhs: DensePrimeField<F>) {
        mul_poly(&mut self.0, &rhs.0, F::ZERO);
    }
}

impl<F: PrimeField> MulAssign<&F> for DensePrimeField<F> {
    fn mul_assign(&mut self, rhs: &F) {
        *self *= *rhs;
    }
}

impl<F: PrimeField> MulAssign<F> for DensePrimeField<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.0.par_iter_mut().for_each(|coeff| *coeff *= rhs);
    }
}

impl<F: PrimeField> Div<&DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn div(self, rhs: &DensePrimeField<F>) -> Self::Output {
        let (quotient, _) = self.poly_mod(rhs);
        quotient
    }
}

impl<F: PrimeField> Div<DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn div(self, rhs: DensePrimeField<F>) -> Self::Output {
        self / &rhs
    }
}

impl<F: PrimeField> Div<&DensePrimeField<F>> for DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn div(self, rhs: &DensePrimeField<F>) -> Self::Output {
        &self / rhs
    }
}

impl<F: PrimeField> Div for DensePrimeField<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<F: PrimeField> DivAssign<&DensePrimeField<F>> for DensePrimeField<F> {
    fn div_assign(&mut self, rhs: &DensePrimeField<F>) {
        *self = &*self / rhs;
    }
}

impl<F: PrimeField> DivAssign<DensePrimeField<F>> for DensePrimeField<F> {
    fn div_assign(&mut self, rhs: DensePrimeField<F>) {
        *self = &*self / &rhs;
    }
}

impl<F: PrimeField> Rem<&DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn rem(self, rhs: &DensePrimeField<F>) -> Self::Output {
        let (_, remainder) = self.poly_mod(rhs);
        remainder
    }
}

impl<F: PrimeField> Rem<DensePrimeField<F>> for &DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn rem(self, rhs: DensePrimeField<F>) -> Self::Output {
        self % &rhs
    }
}

impl<F: PrimeField> Rem<&DensePrimeField<F>> for DensePrimeField<F> {
    type Output = DensePrimeField<F>;

    fn rem(self, rhs: &DensePrimeField<F>) -> Self::Output {
        &self % rhs
    }
}

impl<F: PrimeField> Rem for DensePrimeField<F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        &self % &rhs
    }
}

impl<F: PrimeField> RemAssign<&DensePrimeField<F>> for DensePrimeField<F> {
    fn rem_assign(&mut self, rhs: &DensePrimeField<F>) {
        *self = &*self % rhs;
    }
}

impl<F: PrimeField> RemAssign<DensePrimeField<F>> for DensePrimeField<F> {
    fn rem_assign(&mut self, rhs: DensePrimeField<F>) {
        *self = &*self % &rhs;
    }
}

impl<F: PrimeField> Serialize for DensePrimeField<F> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            self.0
                .par_iter()
                .map(|sc| hex::encode(sc.to_repr().as_ref()))
                .collect::<Vec<_>>()
                .serialize(s)
        } else {
            let mut bytes = vec![];
            for c in self.0.iter() {
                bytes.extend_from_slice(c.to_repr().as_ref());
            }
            s.serialize_bytes(&bytes)
        }
    }
}

impl<'de, F: PrimeField> Deserialize<'de> for DensePrimeField<F> {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if d.is_human_readable() {
            let scalars = <Vec<String>>::deserialize(d)?;
            let mut coeffs = Vec::with_capacity(scalars.len());
            for s in scalars {
                let mut repr = F::Repr::default();
                let bytes = hex::decode(s).map_err(|e| E::custom(format!("Invalid hex: {}", e)))?;
                repr.as_mut().copy_from_slice(&bytes[..]);
                coeffs.push(
                    Option::from(F::from_repr(repr))
                        .ok_or(E::custom("Invalid scalar".to_string()))?,
                );
            }
            Ok(Self(coeffs))
        } else {
            struct PolyVisitor<F: PrimeField>(PhantomData<F>);

            impl<'de, F: PrimeField> Visitor<'de> for PolyVisitor<F> {
                type Value = DensePrimeField<F>;

                fn expecting(&self, f: &mut Formatter<'_>) -> FmtResult {
                    write!(f, "a byte sequence")
                }

                fn visit_bytes<EE>(self, v: &[u8]) -> Result<Self::Value, EE>
                where
                    EE: E,
                {
                    let mut repr = F::Repr::default();
                    let sc_len = repr.as_ref().len();
                    if v.len() % sc_len != 0 {
                        return Err(E::custom(format!("Invalid length: {}", v.len())));
                    }
                    let mut coeffs = Vec::with_capacity(v.len() / sc_len);
                    for chunk in v.chunks(sc_len) {
                        repr.as_mut().copy_from_slice(chunk);
                        coeffs.push(
                            Option::from(F::from_repr(repr))
                                .ok_or(E::custom("Invalid scalar".to_string()))?,
                        );
                    }
                    Ok(DensePrimeField(coeffs))
                }
            }

            d.deserialize_bytes(PolyVisitor(PhantomData))
        }
    }
}

impl<F: PrimeField> Polynomial<F> for DensePrimeField<F> {
    type X = F;

    const ZERO: Self = Self(Vec::new());

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn one() -> Self {
        Self(vec![F::ONE])
    }

    fn degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            debug_assert!(self.0.last().map_or(false, |c| bool::from(!c.is_zero())));
            self.0.len() - 1
        }
    }

    fn evaluate(&self, x: &Self::X) -> F {
        if self.is_zero() {
            F::ZERO
        } else {
            // Horner's method
            self.0.iter().rfold(F::ZERO, move |acc, c| acc * x + c)
        }
    }

    fn is_cyclotomic(&self) -> bool {
        let m_one = -F::ONE;
        for coeff in &self.0 {
            if (!(coeff.ct_eq(&m_one) | coeff.ct_eq(&F::ONE) | coeff.ct_eq(&F::ZERO))).into() {
                return false;
            }
        }
        true
    }

    fn poly_mod(&self, m: &Self) -> (Self, Self) {
        if m.is_cyclotomic() {
            let degree =
                m.0.iter()
                    .rposition(|c| bool::from(!c.is_zero()))
                    .expect("m is cyclotomic");
            return self.poly_mod_cyclotomic(degree);
        }
        let self_degree = self.degree();
        let m_degree = m.degree();
        let mut self_trimmed = self.clone();
        self_trimmed.trim();
        if self_degree < m_degree {
            return (Self::ZERO, self_trimmed);
        }
        debug_assert!(m_degree > 0);

        let largest_coeff_inv = m.0[m_degree].invert().expect("should not be zero");
        let mut coefficients = vec![F::ZERO; self_degree - m_degree + 1];
        let mut remainder = Self(self_trimmed.0.clone());

        let mut degree = remainder.degree();
        while degree >= m_degree {
            let d = degree - m_degree;
            let lc_index = remainder
                .0
                .iter()
                .rposition(|c| bool::from(!c.is_zero()))
                .unwrap_or(remainder.0.len() - 1);
            let c = remainder.0[lc_index] * largest_coeff_inv;

            for i in 0..m_degree {
                remainder.0[i + d] -= c * m.0[i];
            }

            remainder.0.pop();
            remainder.trim();
            coefficients[d] = c;
            degree = remainder.degree();
        }

        (Self(coefficients), remainder)
    }
}

impl<'a, F: PrimeField> FromIterator<&'a F> for DensePrimeField<F> {
    fn from_iter<T: IntoIterator<Item = &'a F>>(iter: T) -> Self {
        let mut inner = Vec::new();
        for coeff in iter {
            inner.push(*coeff);
        }
        Self(inner)
    }
}

impl<F: PrimeField> FromIterator<F> for DensePrimeField<F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<F: PrimeField> From<DensePrimeField<F>> for Vec<u8> {
    fn from(p: DensePrimeField<F>) -> Self {
        Self::from(&p)
    }
}

impl<F: PrimeField> From<&DensePrimeField<F>> for Vec<u8> {
    fn from(p: &DensePrimeField<F>) -> Self {
        let mut bytes = vec![];
        for c in p.0.iter() {
            bytes.extend_from_slice(c.to_repr().as_ref());
        }
        bytes
    }
}

impl<F: PrimeField> TryFrom<Vec<u8>> for DensePrimeField<F> {
    type Error = &'static str;
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(&bytes[..])
    }
}

impl<F: PrimeField> TryFrom<&Vec<u8>> for DensePrimeField<F> {
    type Error = &'static str;

    fn try_from(value: &Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_slice())
    }
}

impl<F: PrimeField> TryFrom<&[u8]> for DensePrimeField<F> {
    type Error = &'static str;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let mut repr = F::Repr::default();
        let sc_len = repr.as_ref().len();
        if bytes.len() % sc_len != 0 {
            return Err("Invalid length");
        }
        let mut coeffs = Vec::with_capacity(bytes.len() / sc_len);
        for chunk in bytes.chunks(sc_len) {
            repr.as_mut().copy_from_slice(chunk);
            coeffs.push(Option::from(F::from_repr(repr)).expect("Invalid scalar"));
        }
        Ok(Self(coeffs))
    }
}

impl<F: PrimeField> TryFrom<Box<[u8]>> for DensePrimeField<F> {
    type Error = &'static str;

    fn try_from(value: Box<[u8]>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_ref())
    }
}

impl<F: PrimeField> DensePrimeField<F> {
    fn poly_mod_cyclotomic(&self, degree: usize) -> (Self, Self) {
        if self.0.len() <= degree {
            return (Self::ZERO, self.clone());
        }

        let mut remainder = self.clone();
        let mut div = Vec::with_capacity(self.0.len() - degree);
        (degree..self.0.len()).rev().for_each(|i| {
            remainder.0[i - degree] += self.0[i];
            div.push(self.0[i]);
        });
        (
            Self(div.into_par_iter().rev().collect()),
            Self(remainder.0.into_par_iter().take(degree).collect()),
        )
    }

    /// Get the polynomial coefficients
    pub fn coefficients(&self) -> &[F] {
        self.0.as_slice()
    }

    /// Get a mutable reference to the polynomial coefficients
    pub fn coefficients_mut(&mut self) -> &mut [F] {
        self.0.as_mut_slice()
    }

    /// Create a polynomial given the coefficients
    pub fn from_coefficients<B: AsRef<[F]>>(coefficients: B) -> Self {
        let mut out = Self(coefficients.as_ref().to_vec());
        out.trim();
        out
    }

    /// Randomly create a polynomial with the specified degree
    pub fn random(degree: usize, mut rng: impl RngCore) -> Self {
        let mut coeffs = Vec::with_capacity(degree);
        for _ in 0..=degree {
            coeffs.push(F::random(&mut rng));
        }
        Self::from_coefficients(coeffs)
    }

    /// Remove leading coefficients that are zero
    pub fn trim(&mut self) {
        for i in (0..self.0.len()).rev() {
            if !bool::from(self.0[i].is_zero()) {
                self.0.truncate(i + 1);
                return;
            }
        }
    }

    /// Compute the dot product of the two polynomials
    pub fn dot_product(&self, other: &Self) -> F {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;
    use rstest::*;

    #[rstest]
    #[case::k256(k256::Scalar::default())]
    #[case::p256(p256::Scalar::default())]
    #[case::p384(p384::Scalar::default())]
    #[case::ed25519(vsss_rs::curve25519::WrappedScalar::default())]
    #[case::bls12_381(bls12_381_plus::Scalar::default())]
    #[case::ed448(ed448_goldilocks_plus::Scalar::default())]
    #[case::jubjub(jubjub::Scalar::default())]
    fn poly_mod<F: PrimeField>(#[case] _f: F) {
        let mut rng = ChaChaRng::from_seed([7u8; 32]);

        let a = DensePrimeField((0..63).map(|_| F::random(&mut rng)).collect());
        let b = DensePrimeField((0..33).map(|_| F::random(&mut rng)).collect());

        let (div, rem) = a.poly_mod(&b);
        let div_b = &div * &b;
        let mut div_b_pr = &div_b + &rem;
        div_b_pr.trim();
        assert_eq!(a, div_b_pr);
    }

    #[rstest]
    #[case::k256(k256::Scalar::default())]
    #[case::p256(p256::Scalar::default())]
    #[case::p384(p384::Scalar::default())]
    #[case::ed25519(vsss_rs::curve25519::WrappedScalar::default())]
    #[case::bls12_381(bls12_381_plus::Scalar::default())]
    #[case::ed448(ed448_goldilocks_plus::Scalar::default())]
    #[case::jubjub(jubjub::Scalar::default())]
    fn poly_mod_cyclotomic<F: PrimeField>(#[case] _f: F) {
        const DEGREE: usize = 320;
        let mut rng = ChaChaRng::from_seed([11u8; 32]);
        let a = DensePrimeField((0..2 * DEGREE - 1).map(|_| F::random(&mut rng)).collect());
        let mut b = DensePrimeField((0..DEGREE + 1).map(|_| F::ZERO).collect());
        b.0[0] = -F::ONE;
        b.0[DEGREE] = F::ONE;
        let (div, rem) = a.poly_mod(&b);
        let div_b = &div * &b;
        let mut div_b_pr = &div_b + &rem;
        div_b_pr.trim();
        assert_eq!(a, div_b_pr);
    }

    #[rstest]
    #[case::k256(k256::Scalar::default())]
    #[case::p256(p256::Scalar::default())]
    #[case::p384(p384::Scalar::default())]
    #[case::ed25519(vsss_rs::curve25519::WrappedScalar::default())]
    #[case::bls12_381(bls12_381_plus::Scalar::default())]
    #[case::ed448(ed448_goldilocks_plus::Scalar::default())]
    #[case::jubjub(jubjub::Scalar::default())]
    fn serialize<F: PrimeField>(#[case] _f: F) {
        let mut rng = ChaChaRng::from_seed([11u8; 32]);
        let a = DensePrimeField((0..10).map(|_| F::random(&mut rng)).collect());
        let res = serde_json::to_string(&a);
        assert!(res.is_ok());
        let serialized = res.unwrap();
        let res = serde_json::from_str::<DensePrimeField<F>>(&serialized);
        assert!(res.is_ok());
        let deserialized = res.unwrap();
        assert_eq!(a, deserialized);

        let res = serde_bare::to_vec(&a);
        assert!(res.is_ok());
        let serialized = res.unwrap();
        let res = serde_bare::from_slice::<DensePrimeField<F>>(&serialized);
        assert!(res.is_ok());
        let deserialized = res.unwrap();
        assert_eq!(a, deserialized);
    }

    #[test]
    fn display() {
        let a = DensePrimeField(vec![
            k256::Scalar::from(5u64),
            k256::Scalar::from(10u64),
            k256::Scalar::from(20u64),
        ]);
        assert_eq!(a.to_string(), "0000000000000000000000000000000000000000000000000000000000000005 + 000000000000000000000000000000000000000000000000000000000000000a² + 0000000000000000000000000000000000000000000000000000000000000014³");
    }

    #[rstest]
    #[case::k256(k256::Scalar::default())]
    #[case::p256(p256::Scalar::default())]
    #[case::p384(p384::Scalar::default())]
    #[case::ed25519(vsss_rs::curve25519::WrappedScalar::default())]
    #[case::bls12_381(bls12_381_plus::Scalar::default())]
    #[case::ed448(ed448_goldilocks_plus::Scalar::default())]
    #[case::jubjub(jubjub::Scalar::default())]
    fn add<F: PrimeField>(#[case] _f: F) {
        let mut rng = ChaChaRng::from_seed([5u8; 32]);
        // Same length
        let a = DensePrimeField::<F>::random(5, &mut rng);
        let b = DensePrimeField::<F>::random(5, &mut rng);

        let c = &a + &b;
        for i in 0..6 {
            assert_eq!(c.0[i], a.0[i] + b.0[i]);
        }

        // Different lengths
        let a = DensePrimeField::<F>::random(4, &mut rng);
        let b = DensePrimeField::<F>::random(7, &mut rng);
        let c = &a + &b;
        assert_eq!(c.0.len(), 8);
        for i in 0..5 {
            assert_eq!(c.0[i], a.0[i] + b.0[i]);
        }
        for i in 5..8 {
            assert_eq!(c.0[i], b.0[i]);
        }
    }

    #[rstest]
    #[case::k256(k256::Scalar::default())]
    #[case::p256(p256::Scalar::default())]
    #[case::p384(p384::Scalar::default())]
    #[case::ed25519(vsss_rs::curve25519::WrappedScalar::default())]
    #[case::bls12_381(bls12_381_plus::Scalar::default())]
    #[case::ed448(ed448_goldilocks_plus::Scalar::default())]
    #[case::jubjub(jubjub::Scalar::default())]
    fn sub<F: PrimeField>(#[case] _f: F) {
        let mut rng = ChaChaRng::from_seed([4u8; 32]);
        // Same length
        let a = DensePrimeField::<F>::random(5, &mut rng);
        let b = DensePrimeField::<F>::random(5, &mut rng);

        let c = &a - &b;
        for i in 0..6 {
            assert_eq!(c.0[i], a.0[i] - b.0[i]);
        }

        // Different lengths
        let a = DensePrimeField::<F>::random(4, &mut rng);
        let b = DensePrimeField::<F>::random(7, &mut rng);
        let c = &a - &b;
        assert_eq!(c.0.len(), 8);
        for i in 0..5 {
            assert_eq!(c.0[i], a.0[i] - b.0[i]);
        }
        for i in 5..8 {
            assert_eq!(c.0[i], -b.0[i]);
        }
    }

    #[test]
    fn dot_product() {
        let a = DensePrimeField(vec![
            k256::Scalar::from(5u64),
            k256::Scalar::from(10u64),
            k256::Scalar::from(20u64),
        ]);
        let b = DensePrimeField(vec![
            k256::Scalar::from(2u64),
            k256::Scalar::from(4u64),
            k256::Scalar::from(8u64),
        ]);

        let c = a.dot_product(&b);
        let expected = k256::Scalar::from(5u64) * k256::Scalar::from(2u64)
            + k256::Scalar::from(10u64) * k256::Scalar::from(4u64)
            + k256::Scalar::from(20u64) * k256::Scalar::from(8u64);
        assert_eq!(c, expected);
    }
}
