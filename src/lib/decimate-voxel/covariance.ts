function covarianceComponentsFromValuesWithScales(
    scaleX: number,
    scaleY: number,
    scaleZ: number,
    quatX: number,
    quatY: number,
    quatZ: number,
    quatW: number,
    out: Float32Array | Float64Array,
    outOffset = 0,
    scalesOut: Float64Array = null,
    scalesOffset = 0
) {
    let x = quatX;
    let y = quatY;
    let z = quatZ;
    let w = quatW;
    const len2 = x * x + y * y + z * z + w * w;
    if (len2 < 1e-20) {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        w = 1.0;
    } else {
        const inv = 1.0 / Math.sqrt(len2);
        x *= inv;
        y *= inv;
        z *= inv;
        w *= inv;
    }

    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;

    const s0 = Math.exp(scaleX);
    const s1 = Math.exp(scaleY);
    const s2 = Math.exp(scaleZ);
    if (scalesOut) {
        scalesOut[scalesOffset + 0] = s0;
        scalesOut[scalesOffset + 1] = s1;
        scalesOut[scalesOffset + 2] = s2;
    }
    const s2x = s0 * s0;
    const s2y = s1 * s1;
    const s2z = s2 * s2;

    const r00 = 1.0 - 2.0 * (yy + zz);
    const r10 = 2.0 * (xy - wz);
    const r20 = 2.0 * (xz + wy);
    const r01 = 2.0 * (xy + wz);
    const r11 = 1.0 - 2.0 * (xx + zz);
    const r21 = 2.0 * (yz - wx);
    const r02 = 2.0 * (xz - wy);
    const r12 = 2.0 * (yz + wx);
    const r22 = 1.0 - 2.0 * (xx + yy);

    out[outOffset + 0] = r00 * r00 * s2x + r10 * r10 * s2y + r20 * r20 * s2z;
    out[outOffset + 1] = r00 * r01 * s2x + r10 * r11 * s2y + r20 * r21 * s2z;
    out[outOffset + 2] = r00 * r02 * s2x + r10 * r12 * s2y + r20 * r22 * s2z;
    out[outOffset + 3] = r01 * r01 * s2x + r11 * r11 * s2y + r21 * r21 * s2z;
    out[outOffset + 4] = r01 * r02 * s2x + r11 * r12 * s2y + r21 * r22 * s2z;
    out[outOffset + 5] = r02 * r02 * s2x + r12 * r12 * s2y + r22 * r22 * s2z;
    return out;
}

function makeCovarianceToScaleQuatScratch() {
    return {
        a: [new Float64Array(3), new Float64Array(3), new Float64Array(3)],
        v: [new Float64Array(3), new Float64Array(3), new Float64Array(3)],
        eigen: [
            { value: 0.0, col: 0 },
            { value: 0.0, col: 1 },
            { value: 0.0, col: 2 }
        ],
        rot: new Float64Array(9),
        quat: new Float64Array(4)
    };
}

function quaternionFromRotationMatrix(
    r00: number,
    r01: number,
    r02: number,
    r10: number,
    r11: number,
    r12: number,
    r20: number,
    r21: number,
    r22: number,
    out: Float64Array
) {
    const trace = r00 + r11 + r22;
    let x;
    let y;
    let z;
    let w;

    if (trace > 0.0) {
        const s = Math.sqrt(trace + 1.0) * 2.0;
        w = 0.25 * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        const s = Math.sqrt(Math.max(1.0 + r00 - r11 - r22, 1e-20)) * 2.0;
        w = (r21 - r12) / s;
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if (r11 > r22) {
        const s = Math.sqrt(Math.max(1.0 + r11 - r00 - r22, 1e-20)) * 2.0;
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
    } else {
        const s = Math.sqrt(Math.max(1.0 + r22 - r00 - r11, 1e-20)) * 2.0;
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
    }

    const len2 = x * x + y * y + z * z + w * w;
    if (len2 < 1e-20) {
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 1.0;
        return out;
    }
    const inv = 1.0 / Math.sqrt(len2);
    out[0] = x * inv;
    out[1] = y * inv;
    out[2] = z * inv;
    out[3] = w * inv;
    return out;
}

function covarianceToScaleQuat(
    c00: number,
    c01: number,
    c02: number,
    c11: number,
    c12: number,
    c22: number,
    scaleLogOut: Float32Array,
    scaleOff: number,
    quatsOut: Float32Array,
    quatOff: number,
    scratch: ReturnType<typeof makeCovarianceToScaleQuatScratch>
) {
    const activeScratch = scratch;
    const a = activeScratch.a;
    a[0][0] = c00;
    a[0][1] = c01;
    a[0][2] = c02;
    a[1][0] = c01;
    a[1][1] = c11;
    a[1][2] = c12;
    a[2][0] = c02;
    a[2][1] = c12;
    a[2][2] = c22;

    const v = activeScratch.v;
    v[0][0] = 1.0;
    v[0][1] = 0.0;
    v[0][2] = 0.0;
    v[1][0] = 0.0;
    v[1][1] = 1.0;
    v[1][2] = 0.0;
    v[2][0] = 0.0;
    v[2][1] = 0.0;
    v[2][2] = 1.0;

    for (let iter = 0; iter < 12; iter++) {
        let p = 0;
        let q = 1;
        let maxOff = Math.abs(a[0][1]);
        const abs02 = Math.abs(a[0][2]);
        if (abs02 > maxOff) {
            p = 0;
            q = 2;
            maxOff = abs02;
        }
        const abs12 = Math.abs(a[1][2]);
        if (abs12 > maxOff) {
            p = 1;
            q = 2;
            maxOff = abs12;
        }

        const scale = Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]) + 1.0;
        if (maxOff <= 1e-10 * scale) {
            break;
        }

        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        if (Math.abs(apq) <= 1e-20) {
            continue;
        }

        const tau = (aqq - app) / (2.0 * apq);
        const signTau = tau >= 0.0 ? 1.0 : -1.0;
        const t = signTau / (Math.abs(tau) + Math.sqrt(1.0 + tau * tau));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        for (let r = 0; r < 3; r++) {
            if (r === p || r === q) {
                continue;
            }
            const arp = a[r][p];
            const arq = a[r][q];
            const nextRp = c * arp - s * arq;
            const nextRq = s * arp + c * arq;
            a[r][p] = nextRp;
            a[p][r] = nextRp;
            a[r][q] = nextRq;
            a[q][r] = nextRq;
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (let r = 0; r < 3; r++) {
            const vrp = v[r][p];
            const vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    const eigen = activeScratch.eigen;
    eigen[0].value = Math.max(a[0][0], 1e-20);
    eigen[0].col = 0;
    eigen[1].value = Math.max(a[1][1], 1e-20);
    eigen[1].col = 1;
    eigen[2].value = Math.max(a[2][2], 1e-20);
    eigen[2].col = 2;
    eigen.sort((lhs, rhs) => rhs.value - lhs.value || lhs.col - rhs.col);

    const rot = activeScratch.rot;
    for (let dstCol = 0; dstCol < 3; dstCol++) {
        const srcCol = eigen[dstCol].col;
        rot[dstCol + 0] = v[0][srcCol];
        rot[dstCol + 3] = v[1][srcCol];
        rot[dstCol + 6] = v[2][srcCol];
    }

    const dot01 = rot[0] * rot[1] + rot[3] * rot[4] + rot[6] * rot[7];
    const dot02 = rot[0] * rot[2] + rot[3] * rot[5] + rot[6] * rot[8];
    const dot12 = rot[1] * rot[2] + rot[4] * rot[5] + rot[7] * rot[8];
    if (Math.abs(dot01) > 1e-10 || Math.abs(dot02) > 1e-10 || Math.abs(dot12) > 1e-10) {
        const c0x = rot[0];
        const c0y = rot[3];
        const c0z = rot[6];
        const len0 = Math.max(Math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z), 1e-20);
        rot[0] /= len0;
        rot[3] /= len0;
        rot[6] /= len0;

        const proj1 = rot[0] * rot[1] + rot[3] * rot[4] + rot[6] * rot[7];
        rot[1] -= proj1 * rot[0];
        rot[4] -= proj1 * rot[3];
        rot[7] -= proj1 * rot[6];
        const len1 = Math.max(Math.sqrt(rot[1] * rot[1] + rot[4] * rot[4] + rot[7] * rot[7]), 1e-20);
        rot[1] /= len1;
        rot[4] /= len1;
        rot[7] /= len1;

        rot[2] = rot[3] * rot[7] - rot[6] * rot[4];
        rot[5] = rot[6] * rot[1] - rot[0] * rot[7];
        rot[8] = rot[0] * rot[4] - rot[3] * rot[1];
    }

    const det =
        rot[0] * (rot[4] * rot[8] - rot[5] * rot[7]) -
        rot[1] * (rot[3] * rot[8] - rot[5] * rot[6]) +
        rot[2] * (rot[3] * rot[7] - rot[4] * rot[6]);
    if (det < 0.0) {
        rot[2] *= -1.0;
        rot[5] *= -1.0;
        rot[8] *= -1.0;
    }

    scaleLogOut[scaleOff + 0] = Math.log(Math.sqrt(eigen[0].value));
    scaleLogOut[scaleOff + 1] = Math.log(Math.sqrt(eigen[1].value));
    scaleLogOut[scaleOff + 2] = Math.log(Math.sqrt(eigen[2].value));

    const quat = quaternionFromRotationMatrix(
        rot[0],
        rot[1],
        rot[2],
        rot[3],
        rot[4],
        rot[5],
        rot[6],
        rot[7],
        rot[8],
        activeScratch.quat
    );
    quatsOut[quatOff + 0] = quat[0];
    quatsOut[quatOff + 1] = quat[1];
    quatsOut[quatOff + 2] = quat[2];
    quatsOut[quatOff + 3] = quat[3];
}

const ellipsoidArea = (x: number, y: number, z: number): number => {
    const p = 1.6075;
    return 4 * Math.PI * Math.pow((Math.pow(x * y, p) + Math.pow(x * z, p) + Math.pow(y * z, p)) / 3, 1 / p);
};

export {
    covarianceComponentsFromValuesWithScales,
    makeCovarianceToScaleQuatScratch,
    covarianceToScaleQuat,
    ellipsoidArea
};
