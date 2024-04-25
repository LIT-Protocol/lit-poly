use super::*;
use crypto_bigint::subtle::ConstantTimeEq;
use crypto_bigint::{
    modular::runtime_mod::{DynResidue, DynResidueParams},
    Encoding, NonZero, RandomMod, Uint, Zero,
};
use rand::CryptoRng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

/// A polynomial over big integers with a custom modulus
#[derive(Clone, PartialEq, Eq)]
pub struct DensePolyDynResidue<const N: usize>(
    /// The coefficients of the polynomial
    pub Vec<DynResidue<N>>,
);

unsafe impl<const N: usize> Send for DensePolyDynResidue<N> {}

unsafe impl<const N: usize> Sync for DensePolyDynResidue<N> {}

impl<const N: usize> Default for DensePolyDynResidue<N> {
    fn default() -> Self {
        Self(Vec::default())
    }
}

impl<const N: usize> Debug for DensePolyDynResidue<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "DensePolyDynResidue({:?})", self.0)
    }
}

impl<const N: usize> Display for DensePolyDynResidue<N>
where
    Uint<N>: Encoding,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let vals = self
            .0
            .iter()
            .enumerate()
            .map(|(power, c)| {
                let c = hex::encode(c.retrieve().to_be_bytes());
                if power == 0 {
                    c
                } else {
                    format!("{}x{}", c, to_super_script_digits(power + 1))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        write!(f, "{}", vals)
    }
}

impl<const N: usize> Add<&DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn add(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        let mut output = self.clone();
        add_poly(&mut output.0, &rhs.0);
        output
    }
}

impl<const N: usize> Add<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn add(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        &self + rhs
    }
}

impl<const N: usize> Add<DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn add(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        self + &rhs
    }
}

impl<const N: usize> Add for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn add(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        &self + &rhs
    }
}

impl<const N: usize> AddAssign<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    fn add_assign(&mut self, rhs: &DensePolyDynResidue<N>) {
        add_poly(&mut self.0, &rhs.0);
    }
}

impl<const N: usize> AddAssign for DensePolyDynResidue<N> {
    fn add_assign(&mut self, rhs: DensePolyDynResidue<N>) {
        add_poly(&mut self.0, &rhs.0);
    }
}

impl<const N: usize> Sub<&DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn sub(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        let mut output = self.clone();
        sub_poly(&mut output.0, &rhs.0);
        output
    }
}

impl<const N: usize> Sub<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn sub(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        &self - rhs
    }
}

impl<const N: usize> Sub<DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn sub(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        self - &rhs
    }
}

impl<const N: usize> Sub for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn sub(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        &self - &rhs
    }
}

impl<const N: usize> SubAssign<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    fn sub_assign(&mut self, rhs: &DensePolyDynResidue<N>) {
        sub_poly(&mut self.0, &rhs.0);
    }
}

impl<const N: usize> SubAssign for DensePolyDynResidue<N> {
    fn sub_assign(&mut self, rhs: DensePolyDynResidue<N>) {
        sub_poly(&mut self.0, &rhs.0);
    }
}

impl<const N: usize> Neg for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn neg(self) -> Self::Output {
        let mut output = self.clone();
        output.0.par_iter_mut().for_each(|c| *c = -*c);
        output
    }
}

impl<const N: usize> Neg for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const N: usize> Mul<&DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        let mut output = DensePolyDynResidue::default();
        mul_poly(
            &mut output.0,
            &rhs.0,
            DynResidue::new(&Uint::<N>::ZERO, *rhs.0[0].params()),
        );
        output
    }
}

impl<const N: usize> Mul<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        &self * rhs
    }
}

impl<const N: usize> Mul<DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        self * &rhs
    }
}

impl<const N: usize> Mul for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        &self * &rhs
    }
}

impl<const N: usize> MulAssign<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: &DensePolyDynResidue<N>) {
        mul_poly(
            &mut self.0,
            &rhs.0,
            DynResidue::new(&Uint::<N>::ZERO, *rhs.0[0].params()),
        );
    }
}

impl<const N: usize> MulAssign for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: DensePolyDynResidue<N>) {
        mul_poly(
            &mut self.0,
            &rhs.0,
            DynResidue::new(&Uint::<N>::ZERO, *rhs.0[0].params()),
        );
    }
}

impl<const N: usize> Mul<&DynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &DynResidue<N>) -> Self::Output {
        self * *rhs
    }
}

impl<const N: usize> Mul<DynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: DynResidue<N>) -> Self::Output {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<const N: usize> Mul<&DynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &DynResidue<N>) -> Self::Output {
        &self * rhs
    }
}

impl<const N: usize> Mul<DynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: DynResidue<N>) -> Self::Output {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<const N: usize> MulAssign<&DynResidue<N>> for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: &DynResidue<N>) {
        self.0.par_iter_mut().for_each(|c| *c = *c * rhs);
    }
}

impl<const N: usize> MulAssign<DynResidue<N>> for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: DynResidue<N>) {
        self.0.par_iter_mut().for_each(|c| *c = *c * rhs);
    }
}

impl<const N: usize> Mul<&Uint<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &Uint<N>) -> Self::Output {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<const N: usize> Mul<Uint<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: Uint<N>) -> Self::Output {
        self * &rhs
    }
}

impl<const N: usize> Mul<&Uint<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: &Uint<N>) -> Self::Output {
        &self * rhs
    }
}

impl<const N: usize> Mul<Uint<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn mul(self, rhs: Uint<N>) -> Self::Output {
        &self * rhs
    }
}

impl<const N: usize> MulAssign<&Uint<N>> for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: &Uint<N>) {
        let r = DynResidue::new(rhs, *self.0[0].params());
        self.0.par_iter_mut().for_each(|c| *c = *c * r);
    }
}

impl<const N: usize> MulAssign<Uint<N>> for DensePolyDynResidue<N> {
    fn mul_assign(&mut self, rhs: Uint<N>) {
        *self *= &rhs;
    }
}

impl<const N: usize> Div<&DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn div(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        let (quotient, _) = self.poly_mod(rhs);
        quotient
    }
}

impl<const N: usize> Div<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn div(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        &self / rhs
    }
}

impl<const N: usize> Div<DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn div(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        self / &rhs
    }
}

impl<const N: usize> Div for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn div(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        &self / &rhs
    }
}

impl<const N: usize> DivAssign<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    fn div_assign(&mut self, rhs: &DensePolyDynResidue<N>) {
        let (quotient, _) = self.poly_mod(rhs);
        *self = quotient;
    }
}

impl<const N: usize> DivAssign for DensePolyDynResidue<N> {
    fn div_assign(&mut self, rhs: Self) {
        *self /= &rhs;
    }
}

impl<const N: usize> Rem<&DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn rem(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        let (_, remainder) = self.poly_mod(rhs);
        remainder
    }
}

impl<const N: usize> Rem<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn rem(self, rhs: &DensePolyDynResidue<N>) -> Self::Output {
        &self % rhs
    }
}

impl<const N: usize> Rem<DensePolyDynResidue<N>> for &DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn rem(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        self % &rhs
    }
}

impl<const N: usize> Rem for DensePolyDynResidue<N> {
    type Output = DensePolyDynResidue<N>;

    fn rem(self, rhs: DensePolyDynResidue<N>) -> Self::Output {
        &self % &rhs
    }
}

impl<const N: usize> RemAssign<&DensePolyDynResidue<N>> for DensePolyDynResidue<N> {
    fn rem_assign(&mut self, rhs: &DensePolyDynResidue<N>) {
        let (_, remainder) = self.poly_mod(rhs);
        *self = remainder;
    }
}

impl<const N: usize> RemAssign for DensePolyDynResidue<N> {
    fn rem_assign(&mut self, rhs: DensePolyDynResidue<N>) {
        *self %= &rhs;
    }
}

impl<const N: usize> Serialize for DensePolyDynResidue<N>
where
    Uint<N>: Encoding,
{
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut values = Vec::with_capacity(self.0.len() + 1);
        values.push(*self.0[0].params().modulus());
        values.extend(self.0.iter().map(|c| c.retrieve()));
        values.serialize(s)
    }
}

impl<'de, const N: usize> Deserialize<'de> for DensePolyDynResidue<N>
where
    Uint<N>: Encoding,
{
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<Uint<N>>::deserialize(d)?;
        let modulus = values[0];
        let params = DynResidueParams::new(&modulus);

        let coeffs = values[1..]
            .iter()
            .map(|c| DynResidue::new(c, params))
            .collect();
        Ok(Self(coeffs))
    }
}

impl<const N: usize> FromIterator<DynResidue<N>> for DensePolyDynResidue<N> {
    fn from_iter<T: IntoIterator<Item = DynResidue<N>>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<const N: usize> From<DensePolyDynResidue<N>> for Vec<u8>
where
    Uint<N>: Encoding,
{
    fn from(value: DensePolyDynResidue<N>) -> Self {
        Self::from(&value)
    }
}

impl<const N: usize> From<&DensePolyDynResidue<N>> for Vec<u8>
where
    Uint<N>: Encoding,
{
    fn from(value: &DensePolyDynResidue<N>) -> Self {
        let mut bytes = vec![];
        bytes.extend_from_slice(value.0[0].params().modulus().to_be_bytes().as_ref());
        for c in &value.0 {
            bytes.extend_from_slice(c.retrieve().to_be_bytes().as_ref());
        }
        bytes
    }
}

impl<const N: usize> TryFrom<Vec<u8>> for DensePolyDynResidue<N> {
    type Error = &'static str;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
}

impl<const N: usize> TryFrom<&Vec<u8>> for DensePolyDynResidue<N> {
    type Error = &'static str;

    fn try_from(bytes: &Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
}

impl<const N: usize> TryFrom<&[u8]> for DensePolyDynResidue<N> {
    type Error = &'static str;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        if bytes.len() % Uint::<N>::BYTES != 0 {
            return Err("Invalid number of bytes");
        }
        let count = (bytes.len() / Uint::<N>::BYTES) - 1;

        let modulus = Uint::<N>::from_be_slice(&bytes[..Uint::<N>::BYTES]);
        let params = DynResidueParams::new(&modulus);
        let mut coeffs = vec![DynResidue::zero(params); count];
        for (i, c) in bytes[Uint::<N>::BYTES..]
            .chunks(Uint::<N>::BYTES)
            .zip(coeffs.iter_mut())
        {
            *c = DynResidue::new(&Uint::<N>::from_be_slice(i), params);
        }
        Ok(Self(coeffs))
    }
}

impl<const N: usize> TryFrom<Box<[u8]>> for DensePolyDynResidue<N> {
    type Error = &'static str;

    fn try_from(bytes: Box<[u8]>) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_ref())
    }
}

impl<const N: usize> DensePolyDynResidue<N> {
    pub const ZERO: Self = Self(Vec::new());

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    pub fn degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            debug_assert!(self
                .0
                .last()
                .map_or(false, |c| bool::from(!c.retrieve().is_zero())));
            self.0.len() - 1
        }
    }

    pub fn evaluate(&self, x: &DynResidue<N>) -> DynResidue<N> {
        if self.is_zero() {
            return DynResidue::zero(*x.params());
        } else {
            // Horner's method
            self.0
                .iter()
                .rfold(DynResidue::zero(*x.params()), |acc, c| acc * x + c)
        }
    }

    pub fn is_cyclotomic(&self) -> bool {
        let one = DynResidue::one(*self.0[0].params());
        let zero = DynResidue::zero(*self.0[0].params());
        let m_one = one.neg();
        for coeff in &self.0 {
            if (!(coeff.ct_eq(&m_one) | coeff.ct_eq(&one) | coeff.ct_eq(&zero))).into() {
                return false;
            }
        }
        true
    }

    pub fn poly_mod(&self, m: &Self) -> (Self, Self) {
        if m.is_cyclotomic() {
            let degree =
                m.0.iter()
                    .rposition(|c| bool::from(!c.retrieve().is_zero()))
                    .expect("m is cyclotomic, so it must have a non-zero coefficient");
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

        let (largest_coeff_inv, was_zero) = m.0[m_degree].invert();
        debug_assert!(bool::from(was_zero));
        let mut coefficients =
            vec![DynResidue::zero(*self.0[0].params()); self_degree - m_degree + 1];
        let mut remainder = Self(self_trimmed.0.clone());
        let mut degree = remainder.degree();
        while degree >= m_degree {
            let d = degree - m_degree;
            let lc_index = remainder
                .0
                .iter()
                .rposition(|c| bool::from(!c.retrieve().is_zero()))
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

    pub fn get_one(params: DynResidueParams<N>) -> Self {
        Self(vec![DynResidue::one(params)])
    }

    /// Get the polynomial coefficients
    pub fn coefficients(&self) -> &[DynResidue<N>] {
        self.0.as_slice()
    }

    /// Get a mutable reference to the polynomial coefficients
    pub fn coefficients_mut(&mut self) -> &mut [DynResidue<N>] {
        self.0.as_mut_slice()
    }

    /// Create a polynomial given the coefficients
    pub fn from_coefficients<B: AsRef<[DynResidue<N>]>>(coefficients: B) -> Self {
        let mut out = Self(coefficients.as_ref().to_vec());
        out.trim();
        out
    }

    /// Randomly create a polynomial with the specified degree
    pub fn random(degree: usize, modulus: Uint<N>, mut rng: impl RngCore + CryptoRng) -> Self {
        let params = DynResidueParams::new(&modulus);
        let mut coeffs = Vec::with_capacity(degree + 1);
        let (m, _) = NonZero::<Uint<N>>::const_new(modulus);
        for _ in 0..=degree {
            let c = Uint::<N>::random_mod(&mut rng, &m);
            coeffs.push(DynResidue::new(&c, params));
        }
        Self(coeffs)
    }

    /// Remove leading coefficients that are zero
    pub fn trim(&mut self) {
        for i in (0..self.0.len()).rev() {
            if bool::from(!self.0[i].retrieve().is_zero()) {
                self.0.truncate(i + 1);
                return;
            }
        }
    }

    /// Compute the dot product of the two polynomials
    pub fn dot_product(&self, other: &Self) -> DynResidue<N> {
        let mut acc = DynResidue::zero(*self.0[0].params());
        for (a, b) in self.0.iter().zip(&other.0) {
            acc += a * b;
        }
        acc
    }
}
