const MDS_FREQ_BLOCK_ONE: [i64; 4] = [16, 8, 16, 4];
const MDS_FREQ_BLOCK_TWO: [(i64, i64); 4] = [(-1, 2), (-1, 1), (4, 8), (2, 1)];
const MDS_FREQ_BLOCK_THREE: [i64; 4] = [-8, 1, 1, 2];

// We use split 3 x 4 FFT transform in order to transform our vectors into the frequency domain.
#[inline(always)]
#[allow(clippy::shadow_unrelated)] pub(crate) fn mds_multiply_freq(state: [u64; 16]) -> [u64; 16] {
    let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15] = state;

    let (u0, u1, u2) = fft4_real([s0, s4, s8, s12]);
    let (u4, u5, u6) = fft4_real([s1, s5, s9, s13]);
    let (u8, u9, u10) = fft4_real([s2, s6, s10, s14]);
    let (u12, u13, u14) = fft4_real([s3, s7, s11, s15]);

    // This where the multiplication in frequency domain is done. More precisely, and with
    // the appropriate permuations in between, the sequence of
    // 3-point FFTs --> multiplication by twiddle factors --> Hadamard multiplication -->
    // 3 point iFFTs --> multiplication by (inverse) twiddle factors
    // is "squashed" into one step composed of the functions "block1", "block2" and "block3".
    // The expressions in the aformentioned functions are the result of explicit computations
    // combined with the Karatsuba trick for the multiplication of Complex numbers.

    let [v0, v4, v8, v12] = block1([u0, u4, u8, u12], MDS_FREQ_BLOCK_ONE);
    let [v1, v5, v9, v13] = block2([u1, u5, u9, u13], MDS_FREQ_BLOCK_TWO);
    let [v2, v6, v10, v14] = block3([u2, u6, u10, u14], MDS_FREQ_BLOCK_THREE);
    // The 4th block is not computed as it is similar to the 2nd one, up to complex conjugation,
    // and is, due to the use of the real FFT and iFFT, redundant.

    let [ss0, ss4, ss8, ss12] = ifft4_real((v0, v1, v2));
    let [ss1, ss5, ss9, ss13] = ifft4_real((v4, v5, v6));
    let [ss2, ss6, ss10, ss14] = ifft4_real((v8, v9, v10));
    let [ss3, ss7, ss11, ss15] = ifft4_real((v12, v13, v14));

    [
        ss0, ss1, ss2, ss3, ss4, ss5, ss6, ss7, ss8, ss9, ss10, ss11, ss12, ss13, ss14, ss15,
    ]
}

// We use the real FFT to avoid redundant computations. See https://www.mdpi.com/2076-3417/12/9/4700
#[inline(always)]
fn fft2_real(x: [u64; 2]) -> [i64; 2] {
    [(x[0] as i64 + x[1] as i64), (x[0] as i64 - x[1] as i64)]
}

#[inline(always)]
fn ifft2_real(y: [i64; 2]) -> [u64; 2] {
    // We avoid divisions by 2 by appropriately scaling the MDS matrix constants.
    [(y[0] + y[1]) as u64, (y[0] - y[1]) as u64]
}

#[inline(always)]
fn fft4_real(x: [u64; 4]) -> (i64, (i64, i64), i64) {
    let [z0, z2] = fft2_real([x[0], x[2]]);
    let [z1, z3] = fft2_real([x[1], x[3]]);
    let y0 = z0 + z1;
    let y1 = (z2, -z3);
    let y2 = z0 - z1;
    (y0, y1, y2)
}

#[inline(always)]
fn ifft4_real(y: (i64, (i64, i64), i64)) -> [u64; 4] {
    // In calculating 'z0' and 'z1', division by 2 is avoided by appropriately scaling
    // the MDS matrix constants.
    let z0 = y.0 + y.2;
    let z1 = y.0 - y.2;
    let z2 = y.1 .0;
    let z3 = -y.1 .1;

    let [x0, x2] = ifft2_real([z0, z2]);
    let [x1, x3] = ifft2_real([z1, z3]);

    [x0, x1, x2, x3]
}

#[inline(always)]
fn block1(x: [i64; 4], y: [i64; 4]) -> [i64; 4] {
    let [x0, x1, x2, x3] = x;
    let [y0, y1, y2, y3] = y;
    let z0 = x0 * y0 + x1 * y3 + x2 * y2 + x3 * y1;
    let z1 = x0 * y1 + x1 * y0 + x2 * y3 + x3 * y2;
    let z2 = x0 * y2 + x1 * y1 + x2 * y0 + x3 * y3;
    let z3 = x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0;

    [z0, z1, z2, z3]
}


#[inline(always)]
#[allow(clippy::shadow_unrelated)] fn block2(x: [(i64, i64); 4], y: [(i64, i64); 4]) -> [(i64, i64); 4] {
    let [(x0r, x0i), (x1r, x1i), (x2r, x2i), (x3r, x3i)] = x;
    let [(y0r, y0i), (y1r, y1i), (y2r, y2i), (y3r, y3i)] = y;
    let x0s = x0r + x0i;
    let x1s = x1r + x1i;
    let x2s = x2r + x2i;
    let x3s = x3r + x3i;
    let y0s = y0r + y0i;
    let y1s = y1r + y1i;
    let y2s = y2r + y2i;
    let y3s = y3r + y3i;

    // Compute x0​y0​−ix1​y3​−ix2​y2​−ix3​y1​ using Karatsuba
    let mut m0 = (x0r * y0r, x0i * y0i);
    let mut m1 = (x1r * y3r, x1i * y3i);
    let mut m2 = (x2r * y2r, x2i * y2i);
    let mut m3 = (x3r * y1r, x3i * y1i);
    let z0r = (m0[0] - m0[1])
        + (x1s * y3s - m1[0] - m1[1])
        + (x2s * y2s - m2[0] - m2[1])
        + (x3s * y1s - m3[0] - m3[1]);
    let z0i = (x0s * y0s - m0[0] - m0[1]) + (-m1[0] + m1[1]) + (-m2[0] + m2[1]) + (-m3[0] + m3[1]);
    let z0 = (z0r, z0i);

    // Compute x0​y1​+x1​y0​−ix2​y3​−ix3​y2​ using Karatsuba
    m0 = (x0r * y1r, x0i * y1i);
    m1 = (x1r * y0r, x1i * y0i);
    m2 = (x2r * y3r, x2i * y3i);
    m3 = (x3r * y2r, x3i * y2i);
    let z1r = (m0[0] - m0[1])
        + (m1[0] - m1[1])
        + (x2s * y3s - m2[0] - m2[1])
        + (x3s * y2s - m3[0] - m3[1]);
    let z1i = (x0s * y1s - m0[0] - m0[1])
        + (x1s * y0s - m1[0] - m1[1])
        + (-m2[0] + m2[1])
        + (-m3[0] + m3[1]);
    let z1 = (z1r, z1i);

    // Compute x0​y2​+x1​y1​+x2​y0​−ix3​y3​​ using Karatsuba
    m0 = (x0r * y2r, x0i * y2i);
    m1 = (x1r * y1r, x1i * y1i);
    m2 = (x2r * y0r, x2i * y0i);
    m3 = (x3r * y3r, x3i * y3i);
    let z2r = (m0[0] - m0[1]) + (m1[0] - m1[1]) + (m2[0] - m2[1]) + (x3s * y3s - m3[0] - m3[1]);
    let z2i = (x0s * y2s - m0[0] - m0[1])
        + (x1s * y1s - m1[0] - m1[1])
        + (x2s * y0s - m2[0] - m2[1])
        + (-m3[0] + m3[1]);
    let z2 = (z2r, z2i);

    // Compute x0​y3​+x1​y2​+x2​y1​+x3​y0​​​ using Karatsuba
    m0 = (x0r * y3r, x0i * y3i);
    m1 = (x1r * y2r, x1i * y2i);
    m2 = (x2r * y1r, x2i * y1i);
    m3 = (x3r * y0r, x3i * y0i);
    let z3r = (m0[0] - m0[1]) + (m1[0] - m1[1]) + (m2[0] - m2[1]) + (m3[0] - m3[1]);
    let z3i = (x0s * y3s - m0[0] - m0[1])
        + (x1s * y2s - m1[0] - m1[1])
        + (x2s * y1s - m2[0] - m2[1])
        + (x3s * y0s - m3[0] - m3[1]);
    let z3 = (z3r, z3i);

    [z0, z1, z2, z3]
}

#[inline(always)]
fn block3(x: [i64; 4], y: [i64; 4]) -> [i64; 4] {
    let [x0, x1, x2, x3] = x;
    let [y0, y1, y2, y3] = y;

    let z0 = x0 * y0 - x1 * y3 - x2 * y2 - x3 * y1;
    let z1 = x0 * y1 + x1 * y0 - x2 * y3 - x3 * y2;
    let z2 = x0 * y2 + x1 * y1 + x2 * y0 - x3 * y3;
    let z3 = x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0;

    [z0, z1, z2, z3]
}
