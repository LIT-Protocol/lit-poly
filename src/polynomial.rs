mod field;

use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Common interface for Univariate and Multivariate polynomials
/// that can be dense or sparse
pub trait Polynomial<T>:
    Sized
    + Clone
    + Debug
    + Default
    + Send
    + Sync
    + PartialEq
    + Eq
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Serialize
    + for<'a> Deserialize<'a>
{
    /// The type of values in the polynomial
    type X: Sized + Clone + Debug + Sync;

    /// The polynomial with degree zero
    const ZERO: Self;

    /// Is this polynomial empty
    fn is_zero(&self) -> bool;

    /// The polynomial with degree one
    fn one() -> Self;
    /// The polynomials total degree
    fn degree(&self) -> usize;
    /// Evaluates `self` at the given `X` returning the result as `F`
    fn evaluate(&self, x: &Self::X) -> T;
    /// Determines if this polynomial cyclotomic
    fn is_cyclotomic(&self) -> bool;
    /// Returns the coefficients of the polynomial
    fn coefficients(&self) -> &[Self::X];
    /// Returns the mutable coefficients of the polynomial
    fn coefficients_mut(&mut self) -> &mut [Self::X];
    /// Create a polynomial from the given coefficients
    fn from_coefficients<B: AsRef<[Self::X]>>(coefficients: B) -> Self;
    /// Create a random polynomial of the given degree
    /// where each coefficient is sampled uniformly at random
    fn random(degree: usize, rng: impl RngCore) -> Self;
    /// self mod (m) = (d,r)  such that self = m*d + r
    /// if the polynomial is cyclotomic then
    /// computes self mod (x^deg - 1).
    fn poly_mod(&self, m: &Self) -> (Self, Self);
}
