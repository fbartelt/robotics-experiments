clear; clc

n = 2;
m = 1;
lambda = blkdiag(eye(n), - eye(m));
size_ = n + m;
delta = 1e-6; % used for nummerical differentiation
% papapa(5)
%% O(n, m) test
H = random_O(n, m);
Hd = random_O(n, m);
% -lambda * skew(H(:, 1), 7)

%% SL(n) test
% lambda = eye(size_);
% H = random_SL(size_);
% Hd = 10*random_SL(size_);

%% Orthogonality condition check
% xi = rand(n+m, 1);
% xid = 1.1*rand(n+m, 1);

dist = distance(H, Hd)
Wq = gen_jac(H, lambda);
Wqd = gen_jac(Hd, lambda);
delDdelq = partial_D(H, Hd, delta, true);
delDdelqd = partial_D(H, Hd, delta, false);

condition = @(alpha) norm(delDdelq * Wq - alpha * delDdelqd * Wqd);
alpha = fminsearch(condition, -1)
cond_ort = norm(delDdelq * Wq - alpha * delDdelqd * Wqd)

%% stats plot
T = 100;
stats = zeros(T, 30);
column_index = 1;
stringlist = {};
for m=0:5
    for n=1:5
        cond_mn = zeros(T, 1);
        lambda = blkdiag(eye(n), - eye(m));
        label = sprintf('O(%d, %d)', n, m);
        stringlist{end + 1} = label;
        disp([n, m])
        for sample=1:T
            H = random_O(n, m);
            Hd = random_O(n, m);
            dist = distance(H, Hd);
            Wq = gen_jac(H, lambda);
            Wqd = gen_jac(Hd, lambda);
            delDdelq = partial_D(H, Hd, delta, true);
            delDdelqd = partial_D(H, Hd, delta, false);
            
            condition = @(alpha) norm(delDdelq * Wq - alpha * delDdelqd * Wqd);
            alpha = fminsearch(condition, -1);
            if norm(dist) < 1e-3
                disp(alpha)
            end
            cond_ort = norm(delDdelq * Wq - alpha * delDdelqd * Wqd);
            cond_mn(sample) = cond_ort;
        end
        stats(:, column_index) = cond_mn';
        column_index = column_index + 1;
    end
end
boxchart(stats)
xlabel('O(n, m)')
ylabel('$$\frac{\partial \hat{D}}{\partial q}W(q) - \alpha\frac{\partial \hat{D}}{\partial q_d}W(q_d)$$','interpreter','latex', 'FontSize', 16)
set(gca,'XTickLabel',stringlist);

function S=skew(omega, r)
    %S = [0 -omega(3) omega(2); omega(3) 0 -omega(1); -omega(2) omega(1) 0]; %3x3 case
    % NxN skew-symmetric:
    % omega is r(r - 1)/2-dimensional, S is r-dimensional 
    omega = omega(:)';
    k = length(omega);
    if r == -1
        r = roots([1 -1 -2*k]); %(r^2 -r -2k)
        r=r(r>=0);
    end
    S = diag(omega(1: r-1), 1);
    el = length(omega) - (r - 1);
    diag_num = 2;
    while el
        i = k-el;
        diag_len = length(diag(S, diag_num));
        if el >= diag_len
            S = S + diag(omega(i+1: i + diag_len), diag_num);
            el = el - diag_len;
        else
            aux = [omega(i+1:end) zeros(1, diag_len - el)];
            S = S + diag(aux, diag_num);
            el = el - el;
        end
        diag_num = diag_num + 1;
    end
    S = S - S';
end

function dist = distance(H, Hd)
%     dist=norm(logm(inv(Hd) * H), "fro")^2;
%     dist=norm(eye(size(H)) - (inv(Hd) * H), "fro")^2;
    dist=norm(Hd - H, "fro")^2;
end

function M=random_O(n, m)
    % generates random matrix in O(n, m) indefinite orthogonal group
    X = blkdiag(eye(n), -eye(m));
    r = n+m;
    omega = rand(r*(r-1)/2, 1);
    M = expm(X * skew(omega, -1));
end

function M=random_SL(size_)
    %generates random matrix in SL(n, R) (invertible with det=1)
    L = tril(rand(size_), -1) + eye(size_);
    U = triu(rand(size_), 1) + eye(size_);
    M = L*U;
end

function W=gen_jac(H, lambda)
    % computes 'jacobian' 
%     W = [-lambda * skew(H(:, 1)); -lambda * skew(H(:, 2)); -lambda * skew(H(:, 3))];
      r = length(lambda);
      k = r*(r - 1)/2;
      E = eye(k);
      W = [];
      for i=1:length(H)
          W_i = [];
          for j=1:length(E)
            W_i = cat(2, W_i, lambda * skew(E(:, j), -1) * H(:, i));
          end
          W = cat(1, W, W_i);
      end

%     W = [];
%     for i=1:length(H)
%         W = cat(1, W, -lambda * skew(H(:, i)));
%     end
end

function delDdelq=partial_D(H, Hd, delta, wrt_first)
    % computes \partial{D}\partial{q} if wrt_frist==true
    % or \partial{D}\partial{qd} if wrt_frist==false
    curr_dist = distance(H, Hd);
    delDdelq = zeros(1, numel(H));
    for i=1:numel(H)
        [row, col] = ind2sub(size(H), i);
        delta_H = zeros(size(H));
        delta_H(row, col) = delta;
        if wrt_first
            next_dist = distance(H + delta_H, Hd);
        else
            next_dist = distance(H, Hd + delta_H);
        end
        delDdelq(i) = (next_dist - curr_dist) / delta;
    end
end

%% UNUSED
function M=random_SO(size_)
    theta_x = 2*pi*rand();
    theta_y = 2*pi*rand();
    theta_z = 2*pi*rand();
    Rx = [1 0 0;0 cos(theta_x) -sin(theta_x); 0 sin(theta_x) cos(theta_x)];
    Ry = [cos(theta_y) 0 sin(theta_y); 0 1 0; -sin(theta_y) 0 cos(theta_y)];
    Rz = [cos(theta_z) -sin(theta_z) 0; sin(theta_z) cos(theta_z) 0; 0 0 1];
    M = Rx * Ry * Rz;
end

function M=random_H(size_)
    % generates random matrix in H(2n+1) (Heisenberg group)
    M = triu(rand(size_), 1) + eye(size_);
end

function M=random_GL(size_)
    % generates random invertible matrix GL(n, R) (invertible)
    L = tril(rand(size_));
    U = triu(rand(size_));
    M = L*U;

    % ensure def pos
    [~, flag] = chol(M);
    if flag
        M = random_GL(size_);
    end
end