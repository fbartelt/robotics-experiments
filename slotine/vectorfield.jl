const _INVHALFPI = 0.63660

function _vector_field(curve, alpha, const_vel)
    vector_size = size(curve, 1)
    return p -> _vector_field_vel(p, curve, alpha, const_vel, vector_size)
end

function _vector_field_vel(p, curve, alpha, const_vel, vector_size)
    vec_n, vec_t, min_dist = _compute_ntd(curve, p)
    fun_g = _INVHALFPI * atan(alpha * min_dist)
    fun_h = sqrt(max(1 - fun_g^2, 0))
    abs_const_vel = abs(const_vel)
    sgn = const_vel / (abs_const_vel + 0.00001)
    result = abs_const_vel * (fun_g * vec_n + sgn * fun_h * vec_t)
    return vec(result)
end

function _compute_ntd(curve, p)
    min_dist = Inf
    ind_min = -1

    pr = deepcopy(p)

    for i in 1:size(curve, 1)
        dist_temp = norm(pr - curve[i, :])
        if dist_temp < min_dist
            min_dist = dist_temp
            ind_min = i
        end
    end

    vec_n = curve[ind_min, :] - pr
    vec_n = vec_n / (norm(vec_n) + 0.0001)

    if ind_min == size(curve, 1)
        vec_t = curve[1, :] - curve[ind_min, :]
    else
        vec_t = curve[ind_min + 1, :] - curve[ind_min, :]
    end

    vec_t = vec_t / (norm(vec_t) + 0.0001)

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