const _INVHALFPI = 0.63660

function _distance_fun(p, R, pd, Rd, beta)
    lin_dist = norm(p - pd)^2
    rot_dist = 0.5 * norm(Rd' * R, 2)^2 
    return lin_dist + (beta)^2 * rot_dist
end

function _divide_conquer(curve_segment, p, R, beta)
    curve_points = curve_segment[1]
    curve_frames = curve_segment[2]
    npoints = size(curve_segment[1], 1)

    if npoints == 1
        return _distance_fun(p, R, curve_points[1, :], curve_frames[1, :, :], beta), 0
    end

    mid_index = div(npoints, 2)
    left_segment = (curve_segment[1][1:mid_index, :], curve_segment[2][1:mid_index, :, :])
    right_segment = (curve_segment[1][mid_index+1:end, :], curve_segment[2][mid_index+1:end, :, :])

    left_dist, left_index = _divide_conquer(left_segment, p, R, beta)
    right_dist, right_index = _divide_conquer(right_segment, p, R, beta)

    if left_dist < right_dist
        return left_dist, left_index
    else
        return right_dist, right_index + mid_index
    end
end

function _vector_field_vel(p, R, curve, kf, vr, wr, beta; store_points=true)
    vec_n, vec_t, min_dist = _compute_ntd(curve, p, R, beta, store_points=store_points)
    fun_g = _INVHALFPI * atan(kf * min_dist)
    fun_h = sqrt(max(1 - fun_g^2, 0))
    sgn = 1
    Lambda = blkdiag(eye(3) * vr, eye(3) * wr)

    return Lambda * (-fun_g * vec_n + sgn * fun_h * vec_t)
end

function _compute_ntd(curve, p, R, beta; store_points=true)
    min_dist = Inf
    ind_min = -1

    pr = deepcopy(p) # row vector to simplify computations
    min_dist, ind_min = _divide_conquer(curve, pr, R, beta)

    vec_n_p = pr - curve[1][ind_min, :]
    Rd = curve[2][ind_min, :, :]
    sigma = skew(Rd[:, 1]) * R[:, 1] + skew(Rd[:, 2]) * R[:, 2] + skew(Rd[:, 3]) * R[:, 3]
    vec_n_p = (vec_n_p / (norm(vec_n_p) + 0.0001))
    sigma = (sigma / (norm(sigma) + 0.0001))
    vec_n = [vec_n_p; sigma]
    Rd = curve[2][ind_min, :, :]

    if ind_min == size(curve[1], 1) - 1
        vec_t_p = curve[1][2, :] - curve[1][ind_min, :]
        vec_t_rot = vee(logm(curve[2][2, :, :] * Rd') / self.dt)
    else
        vec_t_p = curve[1][ind_min + 1, :] - curve[1][ind_min, :]
        vec_t_rot = vee(logm(curve[2][ind_min + 1, :, :] * Rd') / self.dt)
    end

    vec_t_p = (vec_t_p / (norm(vec_t_p) + 0.0001))
    vec_t_rot = (vec_t_rot / (norm(vec_t_rot) + 0.0001))
    vec_t = [vec_t_p; vec_t_rot]

    if store_points
        _add_nearest_point((curve[1][ind_min, :], curve[2][ind_min, :, :]))
    end

    if size(vec_n, 1) != 6 || size(vec_t, 1) != 6
        println("Error in vec_n or vec_t: $(size(vec_n)), $(size(vec_t))")
    end

    return vec_n, vec_t, min_dist
end


function parametric_eq_circle(time::Float64)::Matrix{Float64}
    theta = 0:0.001:2*pi
    c1 = 0.1
    c2 = 0.1
    costheta = cos.(theta)
    sintheta = sin.(theta)
    h0 = 0.5
    return hcat([c1 * costheta[i] for i in 1:length(theta)], [c2 * sintheta[i] for i in 1:length(theta)], h0 * ones(length(theta)))
end

function acceleration(p, curve, alpha, const_vel, vector_size, velocity, dt)
    position = deepcopy(p)
    current_vf = _vector_field_vel(position, curve, alpha, const_vel, vector_size)
    # ∂vf/∂x * v̇
    dvfdx = hcat(
        
            _vector_field_vel(
                position + [dt, 0, 0], curve, alpha, const_vel, vector_size) - current_vf,
            _vector_field_vel(
                position + [0, dt, 0], curve, alpha, const_vel, vector_size) - current_vf,
            _vector_field_vel(
                position + [0, 0, dt], curve, alpha, const_vel, vector_size) - current_vf
        
    ) / dt
    a = dvfdx * velocity
    return a
end