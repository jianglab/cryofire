"""
Tools for dealing with SO(3) group and algebra
Adapted from https://github.com/pimdh/lie-vae
All functions are pytorch-ified
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotmat_to_euler(rotmat):
    """
    rotmat: [..., 3, 3] (numpy)
    output: [..., 3, 3]
    """
    return Rotation.from_matrix(rotmat.swapaxes(-2, -1)).as_euler('zxz')


def direction_to_azimuth_elevation(out_of_planes):
    """
    out_of_planes: [..., 3]
    up: Y
    plane: (Z, X)
    output: ([...], [...]) (azimuth, elevation)
    """
    elevation = np.arcsin(out_of_planes[..., 1])
    azimuth = np.arctan2(out_of_planes[..., 0], out_of_planes[..., 2])
    return azimuth, elevation


def s2s2_to_matrix(v1, v2=None):
    """
    Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.
    """
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3:]
        v1 = v1[..., 0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.cat([e1[..., None, :], e2[..., None, :], e3[..., None, :]], -2)


def euler_to_rotmat(euler):
    """
    euler: [..., 3] (numpy)
    output: [..., 3, 3]
    """
    return Rotation.from_euler('zxz', euler).as_matrix().swapaxes(-2, -1)


def select_predicted_latent(pred_full, activated_paths):
    """
    rots_full: [sym_loss_factor * batch_size, ...]
    activated_paths: [batch_size]
    """
    batch_size = activated_paths.shape[0]
    pred_full = pred_full.reshape(-1, batch_size, *pred_full.shape[1:])
    list_arange = np.arange(batch_size)
    pred = pred_full[activated_paths, list_arange]
    return pred


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def rar_to_matrix(axis,axis_rotate=None,axis_rel='z'):
    batch_size, dim = axis.shape
    device = axis.device
    if axis_rotate is None:
        assert dim == 5
        axis_rotate = axis[:, 3:]
        axisAngle = axis[:, 0:3]

    # axis rotation
    axis_norm = normalize_vector(axis)
    b1 = axis_norm[:, 0]
    b2 = axis_norm[:, 1]
    b3 = axis_norm[:, 2]

    # rotation around the axis
    axis_rotate_norm = normalize_vector(axis_rotate)
    a1 = axis_rotate_norm[:, 0]
    a2 = axis_rotate_norm[:, 1]


    if axis_rel =='z':
        # Check for the specific input (0,0,1) and return identity matrix
        identity_mask = (b1 == 0) & (b2 == 0) & (b3 == 1)
        anti_identity_mask = (b1 == 0) & (b2 == 0) & (b3 == -1)
        identity_matrix = torch.eye(3, device=axis.device).unsqueeze(0).repeat(axis.shape[0], 1, 1)
        anti_identity_matrix = -identity_matrix
        # Calculate matrix elements
        m00 = b3 + (b2 ** 2 * (-1 + b3)) / (-1 + torch.abs(b3) ** 2)
        m01 = -(b1 * b2 * (-1 + b3)) / (-1 + torch.abs(b3) ** 2)
        m02 = b1

        m10 = -(b1 * b2 * (-1 + b3)) / (-1 + torch.abs(b3) ** 2)
        m11 = b3 + (b1 ** 2 * (-1 + b3)) / (-1 + torch.abs(b3) ** 2)
        m12 = b2

        m20 = -b1
        m21 = -b2
        m22 = b3

        # Stack the results to form the output matrix of shape (batch_size, 3, 3)
        M1 = torch.stack([
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1)
        ], dim=1)

        # Replace the output with identity matrix where the input is (0,0,1)
        M1[identity_mask] = identity_matrix[identity_mask]
        M1[anti_identity_mask] = anti_identity_matrix[anti_identity_mask]

    if axis_rel == 'x':
        # Check for the specific input (0,0,1) and return identity matrix
        identity_mask = (b1 == 1) & (b2 == 0) & (b3 == 0)
        anti_identity_mask = (b1 == -1) & (b2 == 0) & (b3 == 0)
        identity_matrix = torch.eye(3, device=axis.device).unsqueeze(0).repeat(axis.shape[0], 1, 1)
        anti_identity_matrix = -identity_matrix
        # Calculate matrix elements
        m00 = b1
        m01 = -b2
        m02 = -b3

        m10 = b2
        m11 = b1 + (b3 ** 2 * (-1 + b1)) / (-1 + torch.abs(b1) ** 2)
        m12 = -(b2 * b3 * (-1 + b1)) / (-1 + torch.abs(b1) ** 2)

        m20 = b3
        m21 = -(b2 * b3 * (-1 + b1)) / (-1 + torch.abs(b1) ** 2)
        m22 = b1 + (b2 ** 2 * (-1 + b1)) / (-1 + torch.abs(b1) ** 2)

        # Stack the results to form the output matrix of shape (batch_size, 3, 3)
        M1 = torch.stack([
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1)
        ], dim=1)

        # Replace the output with identity matrix where the input is (0,0,1)
        M1[identity_mask] = identity_matrix[identity_mask]
        M1[anti_identity_mask] = anti_identity_matrix[anti_identity_mask]

    if axis_rel == 'y':
        # Check for the specific input (0,0,1) and return identity matrix
        identity_mask = (b1 == 0) & (b2 == 1) & (b3 == 0)
        anti_identity_mask = (b1 == 0) & (b2 == -1) & (b3 == 0)
        identity_matrix = torch.eye(3, device=axis.device).unsqueeze(0).repeat(axis.shape[0], 1, 1)
        anti_identity_matrix = -identity_matrix
        # Calculate matrix elements
        m00 = b2 + (b3 ** 2 * (-1 + b2)) / (-1 + torch.abs(b2) ** 2)
        m01 = b1
        m02 = -(b1 * b3 * (-1 + b2)) / (-1 + torch.abs(b2) ** 2)

        m10 = -b1
        m11 = b2
        m12 = -b3

        m20 = -(b1 * b3 * (-1 + b2)) / (-1 + torch.abs(b2) ** 2)
        m21 = b3
        m22 = b2 + (b1 ** 2 * (-1 + b2)) / (-1 + torch.abs(b2) ** 2)

        # Stack the results to form the output matrix of shape (batch_size, 3, 3)
        M1 = torch.stack([
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1)
        ], dim=1)

        # Replace the output with identity matrix where the input is (0,0,1)
        M1[identity_mask] = identity_matrix[identity_mask]
        M1[anti_identity_mask] = anti_identity_matrix[anti_identity_mask]

    ma00 = a1
    ma01 = a2
    ma02 = torch.zeros(batch_size).to(device)
    ma10 = -a2
    ma11 = a1
    ma12 = torch.zeros(batch_size).to(device)
    ma20 = torch.zeros(batch_size).to(device)
    ma21 = torch.zeros(batch_size).to(device)
    ma22 = torch.ones(batch_size).to(device)

    M2 = torch.stack([
        torch.stack([ma00, ma01, ma02], dim=-1),
        torch.stack([ma10, ma11, ma12], dim=-1),
        torch.stack([ma20, ma21, ma22], dim=-1)
    ], dim=1)

    output = torch.bmm(M2,M1)

    return output