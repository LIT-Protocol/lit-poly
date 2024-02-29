/*
    Copyright LIT Protocol . All Rights Reserved.
    SPDX-License-Identifier: FSL-1.1
*/
mod dense;
mod sparse;

pub use dense::DensePolyPrimeField;
pub use sparse::SparsePolyPrimeField;

use elliptic_curve::PrimeField;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Common interface for polynomials
/// that can be dense or sparse.
/// Polynomials are ordered by lowest degree first.
/// It is assumed that positions represent the same
/// coefficients and powers
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
    /// self mod (m) = (d,r)  such that self = m*d + r
    /// if the polynomial is cyclotomic then
    /// computes self mod (x^deg - 1).
    fn poly_mod(&self, m: &Self) -> (Self, Self);
}

const SUPERSCRIPT_DIGITS: [&str; 10] = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];

fn to_super_script_digits(n: usize) -> String {
    n.to_string()
        .chars()
        .map(|c| SUPERSCRIPT_DIGITS[c.to_digit(10).expect("a base 10 digit") as usize])
        .collect()
}

pub(crate) fn add_poly<F: PrimeField>(lhs: &mut Vec<F>, rhs: &[F]) {
    let min_len = core::cmp::min(lhs.len(), rhs.len());

    if lhs.len() == min_len {
        for i in rhs.iter().skip(min_len) {
            lhs.push(*i);
        }
    }
    for (i, item) in rhs[..min_len].iter().enumerate() {
        lhs[i] += item;
    }
}

pub(crate) fn sub_poly<F: PrimeField>(lhs: &mut Vec<F>, rhs: &[F]) {
    let min_len = core::cmp::min(lhs.len(), rhs.len());

    if lhs.len() == min_len {
        for i in rhs.iter().skip(min_len) {
            lhs.push(-*i);
        }
    }

    for (i, item) in rhs[..min_len].iter().enumerate() {
        lhs[i] -= item;
    }
}

pub(crate) fn mul_poly<F: PrimeField>(lhs: &mut Vec<F>, rhs: &[F]) {
    if lhs.is_empty() || rhs.is_empty() {
        lhs.clear();
    } else {
        let orig = lhs.clone();
        for i in &mut *lhs {
            *i = F::ZERO;
        }
        // M + N + 1
        lhs.resize_with(lhs.len() + rhs.len(), || F::ZERO);

        // Calculate product
        for (i, item) in orig.iter().enumerate() {
            for (j, jitem) in rhs.iter().enumerate() {
                lhs[i + j] += *jitem * *item;
            }
        }
    }
}
