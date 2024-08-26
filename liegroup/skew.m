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
%     S(1, end) = omega(end);
    S = S - S';
end