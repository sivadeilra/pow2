//! Represents a non-negative power of 2, by storing its exponent.
//!
//! The `Pow2` type is limited in range to `1 .. 2^255` (inclusive). It is simply a wrapper around
//! `u8`, which stores the exponent.
//!
//! `Pow2` is typically used for aligning integer values, such as I/O buffer sizes, page sizes,
//! etc.
//!
//! Negative powers of 2 are not supported because they have little or no application in managing
//! memory, quick multiplication, etc.
//!
//! `Pow2` can represent values outside of the range of specific integer types. For example,
//! `Pow2::from_exponent(40)` cannot be converted to `u32`. The conversion implementations treat
//! this in a way similar to arithmetic overflow for operations such as `x * y`; the operation is
//! defined, but may fail at runtime.
//!
//! It would be possible to define a family of `Pow2` variants, one for each primitive integer type.
//! However, the ergonomics of that design might be undesireable; it is left for future study.
//!
//! The operators defined for `Pow2` are defined in terms of the value of the power of 2, not the
//! exponent. For example, `Pow2 * Pow2` results in adding the exponents, not in multiplying the
//! exponents.
//!
//! # Examples
//!
//! ```
//! # use pow2::Pow2;
//! const PAGE_SIZE: Pow2 = Pow2::from_exponent(12);
//! assert_eq!(u32::from(PAGE_SIZE), 0x1000);
//! let x: u32 = 0x1eee;
//! assert_eq!(PAGE_SIZE.align_down(x), 0x1000);
//! assert_eq!(PAGE_SIZE.align_up(x), Some(0x2000));
//! assert_eq!(PAGE_SIZE.align_up_unchecked(x), 0x2000);
//! ```
//!
//! The `pow2_const!` macro can be used for writing constant powers of 2, using a value rather than
//! an exponent. Due to current limitations of const fns, the `pow2_const!` macro does not check
//! whether the input is a valid power of 2. If it is not a power of 2, then its behavior is
//! undefined.
//!
//! ```
//! use pow2::{Pow2, pow2_const};
//! const PAGE_SIZE: Pow2 = pow2_const!(0x1000);
//! assert_eq!(PAGE_SIZE.exponent(), 12);
//! ```
//!
//! # Signed examples
//!
//! `Pow2` works with signed types as well as unsigned.
//!
//! ```
//! use pow2::Pow2;
//! const ALIGN4: Pow2 = Pow2::from_exponent(2);
//! assert_eq!(ALIGN4.align_up(4i32), Some(4i32));
//! assert_eq!(ALIGN4.align_up(3i32), Some(4i32));
//! assert_eq!(ALIGN4.align_up(2i32), Some(4i32));
//! assert_eq!(ALIGN4.align_up(1i32), Some(4i32));
//! assert_eq!(ALIGN4.align_up(-0i32), Some(0i32));
//! assert_eq!(ALIGN4.align_up(-1i32), Some(0i32));
//! assert_eq!(ALIGN4.align_up(-2i32), Some(0i32));
//! assert_eq!(ALIGN4.align_up(-3i32), Some(0i32));
//! assert_eq!(ALIGN4.align_up(-4i32), Some(-4i32));
//! assert_eq!(ALIGN4.align_up(-5i32), Some(-4i32));
//! assert_eq!(ALIGN4.align_up(-6i32), Some(-4i32));
//! assert_eq!(ALIGN4.align_up(-7i32), Some(-4i32));
//! assert_eq!(ALIGN4.align_up(-8i32), Some(-8i32));
//! ```
//!
//! # Unsafe code examples
//!
//! This library does not contain unsafe code, but it can be used by unsafe code to operate on
//! pointers. This is because Rust allows safe code to operate on the values of unsafe pointers,
//! but not to dereference them.
//!
//! ```
//! use pow2::{Pow2, pow2_const};
//!
//! const U32_ALIGN: Pow2 = Pow2::align_of::<u32>();
//!
//! let array: [u32; 4] = [ 111, 222, 333, 444 ];
//! let item_0_address: *const u8 = &array[0] as *const u32 as *const u8;
//! let item_1_address: *const u8 = &array[1] as *const u32 as *const u8;
//!
//! assert_eq!(
//!   U32_ALIGN.align_up((item_0_address as usize + 1) as *const u8).unwrap(),
//!   item_1_address
//! );
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(not(test), no_std)]

use core::convert::TryFrom;
use core::ops::{Div, Mul};

/// Represents a non-negative power of 2, by storing its exponent.
///
/// This type is limited in range to `2^0 .. 2^255` (inclusive). It is simply a wrapper around a
/// `u8`, which stores the exponent.
///
/// See module docs for more info.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
#[cfg_attr(feature = "zerocopy", derive(zerocopy::AsBytes, zerocopy::FromBytes))]
pub struct Pow2(u8);

impl Pow2 {
    /// Creates a `Pow2` from an exponent. Note that this function does not check whether the
    /// given exponent is valid for any specific primitive type; many values will be useless.
    /// For example, using `Pow2::from_exponent(250)` will not be useful for any current primitive
    /// type.
    pub const fn from_exponent(exponent: u8) -> Pow2 {
        Pow2(exponent)
    }

    /// Returns the exponent of the `Pow2`.
    pub const fn exponent(self) -> u8 {
        self.0
    }

    /// Returns a `Pow2` value for the memory alignment of a given type `T`.
    ///
    /// ```
    /// use pow2::Pow2;
    /// assert_eq!(Pow2::align_of::<u32>(), Pow2::from_exponent(2));
    /// assert_eq!(Pow2::align_of::<u64>(), Pow2::from_exponent(3));
    /// ```
    pub const fn align_of<T>() -> Pow2 {
        Pow2(core::mem::align_of::<T>().trailing_zeros() as u8)
    }

    /// Multiplies two `Pow2` values by adding their exponents. If the values are out of range,
    /// returns `None`.
    ///
    /// ```
    /// use pow2::Pow2;
    /// assert_eq!(Pow2::from_exponent(2).checked_mul(Pow2::from_exponent(3)), Some(Pow2::from_exponent(5)));
    /// assert_eq!(Pow2::from_exponent(200).checked_mul(Pow2::from_exponent(200)), None);
    /// ```
    pub fn checked_mul(self, other: Pow2) -> Option<Pow2> {
        Some(Pow2(self.0.checked_add(other.0)?))
    }

    /// Divides a `Pow2` value by another `Pow2`. If the divisor is greater than the quotient,
    /// then returns `None`.
    ///
    /// ```
    /// use pow2::Pow2;
    /// assert_eq!(Pow2::from_exponent(4).checked_div(Pow2::from_exponent(1)), Some(Pow2::from_exponent(3)));
    /// assert_eq!(Pow2::from_exponent(4).checked_div(Pow2::from_exponent(8)), None);
    /// ```
    pub fn checked_div(self, other: Pow2) -> Option<Pow2> {
        Some(Pow2(self.0.checked_sub(other.0)?))
    }

    /// Multiplies two `Pow2` values by adding their exponents. If the values are out of range,
    /// then normal arithmetic overflow handling occurs.
    ///
    /// ```
    /// use pow2::Pow2;
    /// assert_eq!(Pow2::from_exponent(4) * Pow2::from_exponent(3), Pow2::from_exponent(7));
    /// ```
    pub const fn mul(self, other: Pow2) -> Pow2 {
        Pow2(self.0 + other.0)
    }

    /// Divides a `Pow2` value by another `Pow2`. If the divisor is greater than the quotient,
    /// then normal arithmetic overflow handling occurs.
    ///
    /// ```
    /// use pow2::Pow2;
    /// assert_eq!(Pow2::from_exponent(12) / Pow2::from_exponent(2), Pow2::from_exponent(10));
    /// ```
    pub const fn div(self, other: Pow2) -> Pow2 {
        Pow2(self.0 - other.0)
    }

    /// Returns `true` if `n` is an integer multiple of `self` (is aligned).
    ///
    /// ```
    /// use pow2::Pow2;
    /// const PAGE_SIZE: Pow2 = Pow2::from_exponent(12);
    /// assert_eq!(PAGE_SIZE.is_aligned(0u32), true);
    /// assert_eq!(PAGE_SIZE.is_aligned(1000u32), false);
    /// assert_eq!(PAGE_SIZE.is_aligned(4095u32), false);
    /// assert_eq!(PAGE_SIZE.is_aligned(4096u32), true);
    /// assert_eq!(PAGE_SIZE.is_aligned(4097u32), false);
    /// assert_eq!(PAGE_SIZE.is_aligned(8192u32), true);
    /// assert_eq!(PAGE_SIZE.is_aligned(core::u32::MAX), false);
    /// ```
    pub fn is_aligned<T: IntPow2>(self, n: T) -> bool {
        n.is_aligned(self)
    }

    /// Returns the smallest multiple of `self` (a power of 2) that is greater than or equal to `n`.
    ///
    /// If the aligned value would overflow the range of `T`, then this function returns `None`.
    ///
    /// ```
    /// use pow2::Pow2;
    /// const PAGE_SIZE: Pow2 = Pow2::from_exponent(12);
    /// assert_eq!(PAGE_SIZE.align_up(5000u32), Some(8192));
    /// assert_eq!(PAGE_SIZE.align_up(core::u32::MAX), None); // would overflow
    /// ```
    pub fn align_up<T: IntPow2>(self, n: T) -> Option<T> {
        n.align_up(self)
    }

    /// Returns the smallest multiple of `self` (a power of 2) that is greater than or equal to `n`.
    ///
    /// If the aligned value would overflow the range of `T`, then normal arithmetic overflow
    /// handling occurs.
    ///
    /// ```
    /// use pow2::Pow2;
    ///
    /// ```
    pub fn align_up_unchecked<T: IntPow2>(self, n: T) -> T {
        n.align_up_unchecked(self)
    }

    /// Returns the largest multiple of `self` (a power of 2) that is less than or equal to `n`.
    pub fn align_down<T: IntPow2>(self, n: T) -> T {
        n.align_down(self)
    }
}

impl Mul<Pow2> for Pow2 {
    type Output = Pow2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Pow2) -> Pow2 {
        Pow2(self.0 + other.0)
    }
}

impl Div<Pow2> for Pow2 {
    type Output = Pow2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Pow2) -> Pow2 {
        Pow2(self.0 - other.0)
    }
}

impl core::fmt::Debug for Pow2 {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(fmt, "2^{}", self.0)
    }
}

#[test]
fn test_debug() {
    assert_eq!(format!("{:?}", Pow2::from_exponent(12)), "2^12");
}

/// A zero-sized error type for reporting failures to convert to a `Pow2`.
#[derive(Clone, Debug)]
pub struct NotPow2;

/// Integers and pointers that can be represented as powers of 2.
pub trait IntPow2: Sized {
    /// Produces a value of `Self` from a power of 2.
    fn from_pow2(p: Pow2) -> Self;

    /// Indicates whether a value of `Self` is a multiple of a given power of 2 (is aligned).
    #[allow(clippy::wrong_self_convention)]
    fn is_aligned(self, p: Pow2) -> bool;

    /// Returns the greatest multiple of `p` that is less than or equal to `Self`.
    fn align_down(self, p: Pow2) -> Self;

    /// Returns the smallest multiple of `p` is that is greater than or equal to `Self`.
    fn align_up_unchecked(self, p: Pow2) -> Self;

    /// Returns the smallest multiple of `p` is that is greater than or equal to `Self`,
    /// or `None` if the value cannot be represented in the range of `Self`.
    fn align_up(self, p: Pow2) -> Option<Self>;
}

/// Integers that can be represented as powers of 2.
pub trait IntOnlyPow2: Sized {
    /// Returns `self` divided by `p`, but rounded up.
    ///
    /// In other words, if `self` is not a multiple of `p` then this returns `self / p + 1`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pow2::{Pow2, IntOnlyPow2};
    /// const PAGE_SIZE: Pow2 = Pow2::from_exponent(12);
    /// assert_eq!(5000u32.div_round_up(PAGE_SIZE), 2);
    /// ```
    fn div_round_up(self, p: Pow2) -> Self;
}

macro_rules! define_conversions {
    ($t:ty) => {
        impl TryFrom<$t> for Pow2 {
            type Error = NotPow2;
            fn try_from(n: $t) -> Result<Pow2, NotPow2> {
                if n != 0 {
                    let bits = n.trailing_zeros();
                    if 1 << bits == n {
                        Ok(Pow2(bits as u8))
                    } else {
                        Err(NotPow2)
                    }
                } else {
                    Err(NotPow2)
                }
            }
        }

        impl From<Pow2> for $t {
            fn from(p: Pow2) -> $t {
                1 << p.exponent()
            }
        }

        /*
        impl Into<$t> for Pow2 {
            fn into(self) -> $t {
                1 << p.exponent()
            }
        }
        */

        impl IntPow2 for $t {
            fn from_pow2(p: Pow2) -> Self {
                (1 as $t) << p.exponent()
            }
            fn is_aligned(self, p: Pow2) -> bool {
                let low_mask: $t = (1 << p.exponent()) - 1;
                (self & low_mask) == 0
            }
            fn align_down(self, p: Pow2) -> Self {
                let low_mask: $t = (1 << p.exponent()) - 1;
                self & !low_mask
            }
            fn align_up_unchecked(self, p: Pow2) -> Self {
                let low_mask: $t = (1 << p.exponent()) - 1;
                (self + low_mask) & !low_mask
            }
            fn align_up(self, p: Pow2) -> Option<Self> {
                let low_mask: $t = (1 << p.exponent()) - 1;
                Some((self.checked_add(low_mask)?) & !low_mask)
            }
        }

        impl IntOnlyPow2 for $t {
            fn div_round_up(self, p: Pow2) -> Self {
                let low_mask: $t = (1 << p.exponent()) - 1;
                let partial = (self & low_mask) != 0;
                (self >> p.exponent()) + (partial as $t)
            }
        }

        impl Div<Pow2> for $t {
            type Output = Self;

            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Pow2) -> Self {
                self >> rhs.exponent()
            }
        }

        impl Mul<Pow2> for $t {
            type Output = Self;

            #[allow(clippy::suspicious_arithmetic_impl)]
            fn mul(self, rhs: Pow2) -> Self {
                self << rhs.exponent()
            }
        }
    };
}

define_conversions!(u8);
define_conversions!(u16);
define_conversions!(u32);
define_conversions!(u64);
define_conversions!(u128);
define_conversions!(usize);

define_conversions!(i8);
define_conversions!(i16);
define_conversions!(i32);
define_conversions!(i64);
define_conversions!(i128);
define_conversions!(isize);

// We can operate on the values of raw pointers in safe code because we are not dereferencing them.

macro_rules! define_pointer_conversions {
    ($T:ident => $t:ty) => {
        impl<T> IntPow2 for $t {
            fn from_pow2(p: Pow2) -> Self {
                usize::from_pow2(p) as Self
            }

            fn is_aligned(self, p: Pow2) -> bool {
                (self as usize).is_aligned(p)
            }

            fn align_down(self, p: Pow2) -> Self {
                (self as usize).align_down(p) as Self
            }

            fn align_up_unchecked(self, p: Pow2) -> Self {
                (self as usize).align_up_unchecked(p) as Self
            }

            fn align_up(self, p: Pow2) -> Option<Self> {
                Some((self as usize).align_up(p)? as Self)
            }
        }
    };
}

define_pointer_conversions!(T => *mut T);
define_pointer_conversions!(T => *const T);

/// A helper macro for defining constant values of `Pow2` using constant inputs.
///
/// This macro has no way to validate its inputs, unfortunately, due to limitations in const fns
/// and traits. You must pass it a valid power of 2; if the input is not a power of 2, then the
/// results are undefined.
///
/// # Example
///
/// ```
/// use pow2::{Pow2, pow2_const};
/// const PAGE_SIZE: Pow2 = pow2_const!(4096);
/// assert_eq!(PAGE_SIZE.exponent(), 12);
/// ```
#[macro_export]
macro_rules! pow2_const {
    ($value:expr) => {
        $crate::Pow2::from_exponent(($value as u64).trailing_zeros() as u8)
    };
}

/// The number of bytes in one kebibyte (1,024).
pub const KIB: Pow2 = Pow2::from_exponent(10);

/// The number of bytes in one mebibyte (1,048,576).
pub const MIB: Pow2 = Pow2::from_exponent(20);

/// The number of bytes in one gibibyte (1,073,741,824).
pub const GIB: Pow2 = Pow2::from_exponent(30);

/// The number of bytes in one tebibyte (1,099,511,627,776).
pub const TIB: Pow2 = Pow2::from_exponent(40);
