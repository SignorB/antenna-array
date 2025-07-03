clear
clc
close all

%% LAB 1.1

% n = [10 20 40 80 160];
% K = zeros(length(n));
% err_rel = zeros(length(n));
% r_norm = zeros(length(n));
% 
% 
% for i=1:length(n)
%     R = ones(n(i), 1);
%     A = -diag(R) + diag(R(1:n(i)-1), -1);
% 
%     A(1,:) = 1;
% 
%     b = zeros(n(i), 1);
%     b(1) = 2;
% 
%     [L, U, P] = lu(A);
% 
%     y = fwsub(L, P*b);
%     x= bksub(U, y);
% 
%     x_ex = 2/n(i) * ones(n(i), 1);
% 
% 
%     K(i) = cond(A); % il condizionamento è circa 29 quindi mi aspeto che 
%     err_rel(i) = norm(x_ex-x) / norm(x_ex);
% 
%     r_norm(i) = norm(b-A*x) / norm(b);
% end
% 
% figure(1)
% semilogy(n, r_norm, n, err_rel, n, K)

%% LAB 1.2
% n = 5;
% H = hilb(n);
% x_ex = ones(n , 1);
% 
% b = H*x_ex;
% 
% R = chol(H);
% 
% y_c = fwsub(R', b);
% x_c = bksub(R, y_c);
% 
% err_rel_chol = norm(x_c-x_ex)/ norm(x_ex);
% 
% [L, U, P] = lu(H);
% 
% y_m = fwsub(L, P*b);
% x_m = bksub(U, y_m);
% P; % si è stata usata permutazione dell righe
% 
% err_rel_meg = norm(x_m-x_ex) / norm(x_ex);
% res_norm_meg = norm(b- H*x_m) / norm(b);
% K = cond(H);
% 
% list_n = 3:15;
% K = zeros(1, length(list_n));
% err_rel = zeros(1, length(list_n));
% res_norm = zeros(1, length(list_n));
% 
% for i=1:length(list_n)
%     n = list_n(i);
%     H = hilb(n);
%     x_ex = ones(n , 1);
% 
%     b = H*x_ex;
% 
% 
%     [L, U, P] = lu(H);
% 
%     y_m = fwsub(L, P*b);
%     x_m = bksub(U, y_m);
%     P; % si è stata usata permutazione dell righe
% 
%     err_rel(i) = norm(x_m-x_ex) / norm(x_ex);
%     res_norm(i)= norm(b- H*x_m) / norm(b);
%     K(i) = cond(H);
% end
% 
% figure(1)
% semilogy(list_n, err_rel, 'r', list_n, res_norm, 'g', list_n, K, 'b')
% legend('err_rel', 'res_norm', 'K')

%% Lab 3.1

% n= 7;
% A = 9 *diag(ones(1, n)) -3 * diag(ones(1, n-1), -1) - 3* diag(ones(1, n-1), 1) + diag(ones(1, n-2), 2) + diag(ones(1, n-2), -2);
% b = [ 7 4 5 5 5 4 7]';
% 
% el_non_nulli = nnz(A);
% % matrice a dominanza diagonale per righe e per colonne => GS e J
% % convergono entrambe
% 
% eig(A); % Simmetrica + Autovalori tutti positivi => SDP
% D = diag(diag(A));
% E = -tril(A, 1);
% F = - triu(A, 1);
% 
% BJ = D^(-1) * (D-A);
% BGS = (D-E)^(-1) * F;
% 
% rhogs = max(abs(eig(BGS))); % < 1 => converge
% rhoj = max(abs(eig(BJ))); % < 1 => converge
% 
% x0 = zeros(n, 1);
% toll = 1e-6;
% nmax= 1000;
% [x,kde] = jacobi(A, b, x0, toll, nmax);
% 
% [xgs, kgs] = gs(A, b, x0, toll, nmax);
% [xj, kj] = jacobi(A, b, x0, toll, nmax);

%% Lab 3.2
% n = 100;
% R1 = ones(n,1);
% R2 = 2*ones(n,1);
% 
% A = diag(-R2) + diag(R1(1:n-1),-1);
% A(1,:)= 1;
% 
% nnz(A); % 298 elementi non nulli
% Bout = spdiags(A);
% 
% D = diag(diag(A));
% Bj = D\(D-A);   % matrice di iterazione di Jacobi
% 
% E=-tril(A,-1);
% F=-triu(A,1);
% Bgs=(D-E)\F; % matrice di iterazione di Gauss-Seidel
% 
% rho_j = max(abs(eig(Bj)))
% rho_gs = max(abs(eig(Bgs)))
% 
% % Condizione di convergenza soddisfatta in entrambi i casi, mi aspetto una
% % velcoità di convergenza maggiore per GS
% 
% b = ones(n, 1);
% b(1) = 2;
% x0 = zeros(n, 1);
% toll= 1e-6;
% nmax= 1000;
% 
% [xgs, kgs] = gs(A, b, x0, toll, nmax);
% [xj, kj] = jacobi(A, b, x0, toll, nmax);
% 
% kgs
% kj
% xgs
% xj


%% Lab 4.1

% n= 7;
% A = diag(9*ones(1, n)) + diag(-3*ones(1, n-1), -1)+ diag(-3*ones(1, n-1), 1) + diag(ones(1, n-2), -2) + diag(ones(1, n-2), 2);
% b = [7 4 5 5 5 4 7]';
% 
% e = eig(A);
% lambda_max = max(e);
% lambda_min = min(e);
% 
% alpha = 2/lambda_max;
% alpha_opt  = 2/(lambda_max + lambda_min);
% 
% toll = 1e-6;
% nmax= 1000;
% x0 = zeros(n, 1);
% [xopt, kopt] = richardson(A, b, x0, alpha_opt, toll, nmax);
% [x, k] = richardson(A, b, x0, alpha-0.1*alpha, toll, nmax);
% 
% kopt
% k

%% Lab 4.2

% n= 50;
% A = diag(4*ones(1, n)) + diag(-1 * ones(1, n-1), 1) + diag(-1* ones(1, n-1), -1) + diag(-1*ones(1, n-2), 2)+ diag(-1*ones(1, n-2), -2);
% b = 0.2* ones(n, 1);
% 
% 
% toll = 1e-5;
% x0 = zeros(n,  1);
% nmax = 10000;
% 
% U = triu(A);
% if (triu(A)' == tril(A))
%     disp("Simmetrica")
% else 
%     error("Non simmetrica")
% end
% 
% if(min(eig(A)) > 0 )
%     disp("definita positiva")
% else 
%     error("Non definita positiva")
% end
% 
% P = diag(2*ones(1, n)) + diag(-1*ones(1, n-1), -1) + diag(-1*ones(1, n-1), 1);
% condiz = max(eig(A)) / min(eig(A));
% alpha_opt = 2 /(max(eig(P\A)) + min(eig(P\A)));
% 
% [x, iter, err] =  graddyn(A, b, x0, nmax, toll);
% 
% [x_p, iter_p, err_p] = gradprec(A, b, P, x0, nmax, toll);
% 
% % figure(1)
% % semilogy(0:iter, err, 0:iter_p, err_p)
% % legend('non cond', 'condizionato')
% 
% [x_r, iter_r, err_r] = richprec(A, b, P, alpha_opt, x0, nmax, toll);
% [x_rn, k_rn]= richardson(A, b, x0, alpha, tol, nmax);
% 
% figure(1)
% plot(0:iter_r, err_r)

%% Lab 4.3
% n = 47;
% 
% 
% T = spdiags(ones(n,1) * [-1,2,-1],-1:1,n,n);
% F = spdiags(ones(n,1)*[1,-4,6,-4,1],-2:2,n,n);
% h= 1/50;
% A = T./ (h^2) + F./(h^4);
% 
% if (triu(A)' == tril(A))
%     disp("Simmetrica")
% else 
%     error("Non simmetrica")
% end
% 
% if(min(eig(A)) > 0 )
%     disp("definita positiva")
% else 
%     error("Non definita positiva")
% end
% 
% condiz = max(eig(A)) / min(eig(A));
% 
% P1 = diag(diag(A));
% P2 = T./(h^2); 
% P3 = chol(A);
% 
% cond(full(P1\A))
% cond(full(P2\A))
% cond(full(P3\A))


%% Lab 5.1
% f =@(x) exp(x) - x.^2 - sin(x) -1;
% df =@(x) exp(x) -2*x - cos(x);
% 
% toll = 1e-6;
% nmax = 1000;
% x0 = 0.1;
% x02 = 1.2;
% [xvect0, it0] = newton(x0, nmax, toll, f, df);
% [xvect1, it1] = newton(x02, nmax, toll, f, df);
% 
% 
% [p0, c0] = stimap(xvect0);
% [p1, c1] = stimap(xvect1);
% x1 = 0;
% x2= 1.279701331000996;
% 
% err0 = abs(xvect0 - x1);
% err1 = abs(xvect1 - x2);
% 
% figure(1)
% semilogy(0:it0, err0, 0:it1, err1)

%% Lab 5.2

% toll = 1e-6;
% nmax = 1000;
% x0= 0.3;
% 
% phi =@(x) x - cos(x);
% 
% [xvect, it] = ptofis(x0, nmax, toll, phi);


%% Lab 5.3

% toll = 1e-6;
% x0 =[1;0];
% nmax = 1000;
% 
% f =@(x) [ -x(1)+3*exp(3*x(2))-1; -x(1)+x(1).*x(2).^2+2 ];
% 
% J = @(x) [ -1, 3*exp(3*x(2));
%            -1+x(2).^2, 2.*x(1).*x(2) ];
% 
% 
% [xvect, it] = newtonsys(f, J, x0, toll, nmax)

%% Lab 5.4

% f =@(x) (x-1).^2 .* log(x)-(x-1);
% df =@(x) 2*(x-1) .*log(x)+ (x-1).^2 ./x-1;
% 
% a= 0.3;
% b= 2.2;
% xx = linspace(a, b, 1000);
% yy = f(xx);
% 
% phi =@(x) (x-1).^2 .* log(x) + 1;
% 
% 
% x0= 0.7;
% nmax = 1000;
% toll = 1e-5;
% 
% [xvect, it] = ptofis(x0, nmax, toll, phi);
% [xvect_n, it_n]= newton(x0, nmax, toll, f, df);
% 
% stimap(xvect_n)
% stimap(xvect)

%% Lab 6.1

% a = -5;
% b= 5;
% 
% f=@(x) 1 ./ (1+x.^2);
% 
% xx = linspace(a, b, 1000);
% yy = f(xx);
% 
% n = 5;
% x_val = linspace(a, b, n+1);
% 
% P = polyfit(x_val, f(x_val), n);
% yy_val = polyval(P, xx);
% 
% err = norm( xx-yy_val, 'inf');
% 
% figure(1)
% hold on
% plot(xx, yy_val, '--r', xx, yy, '-g')
% 
% n=10;
% x_val = linspace(a, b, n+1);
% 
% P = polyfit(x_val, f(x_val), n);
% yy_val = polyval(P, xx);
% 
% err = norm( xx-yy_val, 'inf');
% 
% 
% plot(xx, yy_val, '--b')
% legend('n=5', 'real', 'n=10')

%% Lab 6.2
% S= [0.00, 0.06, 0.14, 0.25, 0.31, 0.47, 0.60, 0.70];
% E= [0.00, 0.08, 0.14, 0.20, 0.23, 0.25, 0.28, 0.29];
% 
% a = min(S);
% b = max(S);
% xx = linspace(a, b, 1000);
% 
% n = length(S);
% 
% P = polyfit(S, E, n-1);
% yy_l = polyval(P, xx);
% 
% 
% yy_c = interp1(S, E, xx);
% 
% figure(1)
% plot(xx, yy_l, xx, yy_c, S, E, '-o')
% legend('Lagrange', 'Composita', 'S/E')

%% Lab 6.3

% f=@(x)x.* sin(x);
% 
% a = -2;
% b= 6;
% xx = a:0.01:b;
% yy = f(xx);
% n=[2,4,6];
% err = zeros(1, length(n));
% y = zeros(length(n), length(xx));
% err_y = zeros(length(n), length(xx));
% 
% for i=1:length(n)
%     N = n(i);
%     x_dis = linspace(a, b, N+1);
%     P = polyfit(x_dis, f(x_dis), N);
%     y(i,:) = polyval(P, xx);
%     err_y = abs(yy-y(i, :));
%     err(i) = norm(yy-y(i,:) ,'inf');
% end
% 
% figure(1)
% plot(xx, y(1,:), '-r', xx, y(2,:),'-b', xx, y(3,:), '-g')
% legend("n=2", "n=4", "n=6")
% err

%% Lab 6.4

% f=@(x) 1./(1+x.^2);
% a = -5;
% b = 5;
% 
% n=5;
% xx = linspace(a,b, 1000);
% yy = f(xx);
% 
% x_c = -cos(pi .* (0:n)./n);
% 
% x_dis = (a+b)/2 + (b-a)/2 .* x_c;
% y_dis = f(x_dis);
% 
% P = polyfit(x_dis, y_dis, n);
% yy_5 = polyval(P, xx);
% 
% err = norm(yy-yy_5, 'inf')
% 
% n=10;
% xx = linspace(a,b, 1000);
% yy = f(xx);
% 
% x_c = -cos(pi .* (0:n)./n);
% 
% x_dis = (a+b)/2 + (b-a)/2 .* x_c;
% y_dis = f(x_dis);
% 
% P = polyfit(x_dis, y_dis, n);
% yy_5 = polyval(P, xx);
% 
% err = norm(yy-yy_5, 'inf')

%% Lab 7.1


% S= [0.00, 0.06, 0.14, 0.25, 0.31, 0.47, 0.60, 0.70];
% E= [0.00, 0.08, 0.14, 0.20, 0.23, 0.25, 0.28, 0.29];
% 
% xx = linspace(min(S), max(S), 1000);
% 
% 
% P = polyfit(S, E, length(E)-1);
% xx_l = polyval(P, xx);
% 
% 
% x_CMP = interp1(S, E, xx);
% 
% 
% P1 = polyfit(S, E, 1);
% P2 = polyfit(S, E, 2);
% P4 = polyfit(S, E, 4);
% 
% xx_1 = polyval(P1, xx);
% xx_2 = polyval(P2, xx);
% xx_4 = polyval(P4, xx);
% 
% 
% figure(1)
% plot(S, E, '-o', xx, xx_l, xx, xx_4, xx, x_CMP)
% legend("Data", "Lagrange", "n=4", "CMP")
% 
% 
% polyval(P, 0.75)
% polyval(P, 0.40)
% polyval(P1, 0.75)
% polyval(P1, 0.40)
% polyval(P2, 0.75)
% polyval(P2, 0.40)
% polyval(P4, 0.75)
% polyval(P4, 0.40)
% interp1(S, E, [0.75 0.4])

%% Lab 7.3

% f=@(x) x./(2*pi) .* sin(x);
% a = 0;
% b = 2*pi;
% M = 1:20;
% 
% 
% pmed = zeros(1, length(M));
% trap = zeros(1, length(M));
% sim = zeros(1, length(M));
% 
% for i=1:length(M)
%     m = M(i);
%     pmed(i) = pmedcomp(a, b, m, f);
%     trap(i) = trapcomp(a, b, m, f);
%     sim(i) = simpcomp(a, b, m, f);
% end
% 
% % figure(1)
% % plot(M, pmed, M, trap, M, sim);
% % legend("Pmed", 'Trap', "Sim")
% 
% ex = -1 .*ones(1, length(M));
% 
% err_pmed = abs(ex - pmed);
% err_trap =abs(ex - trap);
% err_sim = abs(ex - sim);
% 
% M = (b-a)./M;
% figure(1)
% loglog(M, err_sim, M, err_pmed, M, err_trap)

%% TdE
% 
% fun = @(x) sin(10*pi*x)./(1+10*x.^2);
% 
% a = -3;
% b = 3;
% 
% % y_ris = zeros(length(n), length(xx));
% % ris = zeros(1, length(n));
% % 
% % for i=1:length(n)
% %     N = n(i);
% %     x_dis = linspace(a, b, N+1);
% %     P = polyfit(x_dis, f(x_dis), N);
% %     y_ris(i, :)= polyval(P, xx);
% %     ris(i) = polyval(P, x_signed);
% % end
% % 
% % 
% % 
% % 
% % pbar =[-9.0879e-17, 6.3767e-16, -0.15, 0.2095];
% % 
% % abs(f(x_signed) * ones(1, length(n)) - ris)
% 
% m=2;
% n=10;
% x_nod = linspace (a, b, n + 1);
% f_nod = fun (x_nod);
% P = polyfit (x_nod, f_nod, m);
% x_dis = [a:0.001:b];
% poly_dis = polyval (P, x_dis);
% plot (x_nod, f_nod, 'bo', x_dis, poly_dis, 'k', 'linewidth', 2);

%% TdE
% A = [2, -1 0; -1, 3, 1; 0, -1, 4];
% x_ex = [1/2, 1/3, 1/4]';
% 
% b = A*x_ex;
% 
% x0 = zeros(1, length(x_ex))';
% nmax = 100;
% 
% [xj1, kj1] = jacobi(A, b, x0, 1e-2, nmax)
% [xj2, kj2]= jacobi(A, b, x0, 1e-3, nmax)
% [xj3, kj3]= jacobi(A, b, x0, 1e-4, nmax)
% 
% 
% [xgs, kgs] = gs(A, b, x0, 1e-4, nmax)
% 
% D = diag(diag(A));
% E = -tril(A, -1);
% F = -triu(A, 1);
% 
% Bj = D^-1 *(D-A);
% Bgs = (D-E)^-1 *F;
% 
% max(abs(eig(Bj)))
% max(abs(eig(Bgs)))

%% TdE

% f1=@(x) (4*cos(pi/4 *x) - exp(-4*x));
% f=@(x) -2*x+3;
% 
% a= -1;
% b= 1;
% 
% n= [5, 10, 20, 40];
% 
% h= (b-a) ./n;
% 
% ris = zeros(1, length(n));
% ris1 = zeros(1, length(n));
% 
% 
% for i=1:length(n)
%  ris(i) = trapcomp(a, b, n(i), f);
%  ris1(i) = trapcomp(a, b, n(i), f1);
% end
% 
% ris
% ris1
% 
% I_ex1 =  32/pi * sin(pi/4) - 2 * sinh(4)/4
% I_ex = 6.0;
% 
% err = abs(I_ex * ones(1, length(n)) - ris)
% err1= abs(I_ex1 * ones(1, length(n)) - ris1)
% figure(1)
% loglog(h, err, h, h.^2);

%% TdE



% [L, U, P] = lu(A);



% y = fwsub(L, P*b);
% x = bksub(U, y);

% n=[20, 30, 40];
% err = zeros(1, length(n));
% 
% for i=1:length(n)
%     A = diag(ones(1, n(i)));
%     A(:,n(i)) = ones(1, n(i));
%     A(n(i), :) = [1.1, ones(1, n(i)-1)];
%     b = zeros(n(i), 1);
%     b(1) = 1;
% 
%     [L, U, P ] = lu(A);
% 
%     a= 1.1;
%     x_ex = [1-a/(a+n(i)-3);-a/(a+n(i)-3)*ones(n(i)-2,1);a/(a+n(i)-3)];
%     y = fwsub(L, P*b);
%     x = bksub(U, y);
% 
%     err(i) = norm(x-x_ex) / norm(x_ex);
% end
% 
% err

%% TdE
% 
% f=@(x) cos(7*x) .* exp(-2*x);
% df =@(x) -7*sin(7*x) .* exp(-2*x)+ cos(7*x) .* -2 .* exp(-2* x);
% 
% 
% g =@(x) (cos(7*x)).^4 .* exp(-2*x);
% dg =@(x) 4*(cos(7*x)).^3 .* -7 .* sin(7*x) .* exp(-2*x) -2 .* (cos(7*x)).^4 .* exp(-2*x);
% 
% 
% x0 = -2;
% 
% toll = 1e-6;
% nmax = 1000;

% [xvect, it] = newton(x0, nmax, toll, f, df);
% xvect
% it
% 
% [p, c ] = stimap(xvect)
% 
% x0= -4;
% 
% 
% toll = 1e-6;
% nmax = 1000;
% 
% [xvect, it] = newton(x0, nmax, toll, f, df);
% xvect 
% it

% [xvect, it] = newton(x0, nmax, toll, g, dg);
% dg(xvect(end))
% [p, c] = stimap(xvect);

%% TdE
% f1=@(x) 3* sin(pi/3 .* x)- exp(-3*x);
% f =@(x) -3*x+7;
% 
% a = 0;
% b = 1;
% N = [8, 16, 32, 64];
% h = (b-a) ./N;
% 
% int = zeros(1, length(N));
% int_ex1 = (9./ (2*pi) + (exp(-3)-1) ./3 ) * ones(1, length(N));
% int_ex = 11/2;
% 
% for i=1:length(N) 
%     n= N(i);
%     int(i) = trapcomp(a, b, n, f);
% end
% 
% err = abs(int_ex - int);
% 
% figure(1)
% loglog(h, err, h, h.^2)

%% Lab 9.4
% 
% l1 = -100;
% l2 = -1;
% A = [0, 1; -l1*l2, l1+l2];
% f =@(t ,y) A*y;
% y0 = [1;1];
% a = 0;
% b = 5;
% h = 1e-4;
% [t_h, u] = eulero_avanti_sistemi(f, [a, b], y0, (b-a)/h );
% figure(1)
% plot(t_h, u(1,:), t_h, u(2,:))
% h_max = 2 ./ max(abs(eig(A)));
% 
% [t45, u45] = ode45(f, [a, b], y0);
% 
% figure(2)
% plot(t45, u45)
% it45 = length(t45)
% h45 = diff(t45);
% 
% 
% [t15, u15] = ode15s(f, [a, b], y0);
% 
% figure(3)
% plot(t15, u15)
% it15 = length(t15)
% h15 = diff(t15);


%% Lab 9.1

% f=@(t, y) cos(2*y);
% y0 = 0;
% x0 = 0;
% 
% a = 0;
% b = 6;
% 
% y =@(t) 1/2 .* asin((exp(4.*t)-1) ./ (exp(4.*t)+1));
% 
% xx = linspace(a, b, 1000);
% 
% h = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125];
% err = zeros(1, length(h));
% 
% for i=1:length(h)
%     [t_ea, y_ea] = eulero_avanti(f, a,b, y0, h(i));
%     err(i) = max(abs(y(t_ea) - y_ea));
% end
% 
% figure(1)
% loglog(h, err, h, h)


%% Lab 9.2

% lambda = -2;
% f =@(t, y) lambda .* y;
% df =@(t, y) lambda;
% 
% T= 10;
% y0 = 1;
% t0 = 0;
% h = 1.1;
% 
% y =@(t) y0 .* exp(lambda .* t);
% 
% [t_a, u_a] = eulero_avanti(f, t0, T, y0, h);
% [t_i, u_i] = eulero_indietro(f, df, t0, T, y0, h);
% 
% tt = linspace(t0, T,1000);
% yy = y(tt);
% 
% figure(1)
% plot(tt, yy, '-b', t_a, u_a, '--go', t_i, u_i, '--ro')
% legend("Ex", 'A', 'I')

%% Lab 9.3

% f=@(t, y) -y .* (3+2/(2*t+1).^2);
% 
% t0 = 0;
% T = 10;
% y0 = 10*exp(1);
% 
% y =@(t) 10 .* exp(1./(2*t+1)-3*t);
% 
% tt= linspace(t0, T, 1000);
% yy = y(tt);
% 
% h = 1.0;
% 
% err = zeros(1, 4);
% H=[0.1, 0.05, 0.025, 0.0125];
% for i=1:4
%     h =H(i);
%     [ta, ua] = eulero_avanti(f, t0, T, y0, h);
%     err(i) = norm(y(ta)- ua, 'inf');
% end
% 
% figure(1)
% loglog(H, err, H, H)

%% TdE

% f =@(t, y) -y .* (2+ 1./(t+1).^2);
% df =@(t, y) -(2 + 1 ./ (t + 1).^2);
% 
% y0 = 10*exp(1);
% t0=0;
% T = 10;
% 
% y =@(t) 10 .* exp((1./(t+1)) - 2*t);
% 
% tt = linspace(t0, T, 1000);
% yy = y(tt);
% 
% % H = [0.1, 0.05, 0.025, 0.0125];
% % err = zeros(1, length(H));
% % for i =1:length(H)
% %     h = H(i);
% %     [tc, uc] = crank_nicolson(f, df, t0, T, y0, h);
% %     err(i) = norm(y(tc) - uc ,'inf');
% % end
% % 
% 
% h= 1.0;
% 
% [ta, ua] = eulero_avanti(f, t0, T, y0, h);
% h= 0.1;
% [ta1, ua1] = eulero_avanti(f, t0, T, y0, h);
% 
% figure(1)
% plot(tt, yy, '-b', ta, ua, '--ro', ta1, ua1, '--go')
% legend('ex', 'EA h=1', 'EA h=0.1')
% 

%% TdE
% 
% f =@(x) sin(10.*pi.*x)./(1+10.*x.^2);
% a =-3;
% b = 3;
% n =[6, 8, 10, 14];
% h = (b-a)./n;
% 
% x_s = 1- pi/10;
% xx = linspace(a, b, 10000);
% yy = f(xx);
% 
% val = zeros(1, length(n));
% err = zeros(1, length(n));
% for i=1:length(n)
%     x_dis = linspace(a, b, n(i)+1);
%     y_dis = f(x_dis);
% 
%     P = polyfit(x_dis, y_dis, n(i));
%     val(i) = polyval(P, x_s);
%     err (i) = abs(f(x_s) - val(i));
% end
% 
% 
% val
% err
% 
% n=10;
% k = 2;
% x_dis = linspace(a, b, n+1);
% P = polyfit(x_dis, f(x_dis), k);
% y_dis = polyval(P, xx);
% 
% figure(1)
% plot(x_dis, f(x_dis), 'bo', xx, y_dis, '-k')
% legend('y=f(x)', 'MQ k=2')
% 
%% TdE

% n=7;
% A = diag(ones(1, n));
% A(:, n) = ones(n, 1);
% A(n, :) = 1:n;
% 
% [L, U, P] = lu(A);
% 
% b = [1, zeros(1, n-1)];
% 
% y = fwsub(L, P*b');
% x = bksub(U, y)


% for n=[20, 30, 40]
%     A = diag(ones(1, n));
%     A(:, n) = ones(n, 1);
%     A(n, :)= [1.1, ones(1, n-1)];
% 
%     b = [1; zeros(n-1, 1)];
% 
%     x_ex =[1-(1.1)/(1.1+n-3), -ones(1, n-2) .*(1.1)/(1.1+n-3), (1.1)/(1.1+n-3)]';
% 
%     [L, U, P] = lu(A);
%     y = fwsub(L, P*b);
%     x = bksub(U, y);
% 
%     err = norm(x_ex-x)/norm(x_ex)
% end

%% TdE

f=@(t, y) -1./t .*(2*y+ t.^2.*y.^2);
t0 =1;
T = 5;
y0=1;

y =@(t) 1./(t.^2.*(1+log(t)));

% h=0.2;
% [ta1, ua1] = eulero_avanti(f, t0, T, y0, h);
% 
% h=0.1;
% [ta, ua] = eulero_avanti(f, t0, T, y0, h);
% 
% figure(1)
% plot(ta, y(ta), ta, ua, ta1, ua1)
% legend('f(t)', 'EA h=0.1', 'EA h=0.2')

% H = [0.1, 0.05, 0.025, 0.0125];
% 
% err = zeros(1, length(H));
% for i=1:length(H)
%     h = H(i);
%     [ta, ua] = eulero_avanti(f, t0, T, y0, h);
%     err(i) = norm( y(ta)-ua ,'inf');
% end
% 
% figure(1)
% loglog(H, err, H, H)

%% TdE
% A= [2, -1, 0;-1, 3, 1; 0, -1, 4];
% 
% x_ex = [1/2; 1/3; 1/4];
% 
% x0 = zeros(length(x_ex), 1);
% nmax= 100;
% 
% b = A*x_ex;
% 
% tol = [1e-2, 1e-3, 1e-4];
% 
% for i=1:length(tol)
%     [xj, itj] = jacobi(A, b, x0, tol(i), nmax)
% end
% 
% [xgs, itgs] = gs(A, b, x0, 1e-4, nmax)
% 
% D = diag(diag(A));
% E = -tril(A, -1);
% F = -triu(A, 1);
% 
% Bj = eye(length(A)) - D^-1 * A;
% Bgs = (D-E) \ F;
% 
% rhoj = max(abs(eig(Bj)))
% rhogs = max(abs(eig(Bgs)))

%% TdE

% A =[5, -3, -1; -3, 4, -1; -1, -1, 2];
% x = [1;1;1];
% 
% b = A*x;
% 
% D = diag(diag(A));
% E = - tril(A, -1);
% F = - triu(A, 1);
% 
% Bj = eye(3) - D\A;
% Bgs = (D-E)\F;
% 
% rhoj = max(abs(eig(Bj)));
% rhogs = max(abs(eig(Bgs)));
% 
% toll = 1e-3;
% nmax = 100;
% x0 = [0,0,0]';
% [xj, itj] = jacobi(A, b, x0, toll, nmax)
% [xgs, itgs] = gs(A, b, x0, toll, nmax)

%% TdE
% 
% f =@(x) exp(x/2) + cos(pi/2 * x);
% value = 4 * sinh(1/2) + 4/pi;
% 
% a= -1;
% b= 1;
% n = [4, 8, 16, 32];
% I = zeros(1, length(n));
% 
% for i=1:length(n)
%     I(i) = pmedcomp(a,b,n(i), f);
% end
% 
% h= (b-a)./n;
% err = abs(value* ones(1, length(n)) - I);
% figure(1)
% loglog(h, err,'-ro', h, h.^2, '--r')
% legend("Apprx", "quadratic")
% 
% f =@(x) 9*x+77;
% 
% for i=1:length(n)
%     I(i) = pmedcomp(a,b,n(i), f);
% end
% 
% I

%% TdE
% 
% f =@(x) (x-2) .* exp(x-1);
% alpha = 2;
% 
% phi =@(x) x- ( (x-2) .* exp(x-2)) ./ (2*exp(1) -1);

% xx = linspace(alpha-2, alpha +2, 1000);
% yy = phi(xx);
% 
% figure(1)
% plot(xx, yy, '--r', alpha, phi(alpha), 'ko')

% x0 = 1.5;
% toll = 1e-4;
% nmax = 1000;
% 
% [xvec, it] = ptofis(x0, nmax, toll, phi)
% 
% abs(xvec(it) - xvec(it-1))
% 
% stimap(xvec);

%% TdE

% f =@(t, y) y .* ( (pi * cos(pi*t))./(2+ sin(pi*t)) -1 /2);
% df = @(t, y) (pi * cos(pi*t))./(2+ sin(pi*t)) -1 /2;
% t0 = 0;
% T = 10;
% y0 = 2;
% 
% y =@(t) (2+ sin(pi*t)).* exp(-t/2);
% 
% xx = linspace(t0, T, 1000);
% yy = y(xx);
% % 
% % figure(1)
% % plot(xx, yy, '--r', tei, uei, '--k', tei1, uei1, '--g')
% 
% 
% H = [0.04, 0.02, 0.01, 0.005, 0.0025];
% err = zeros(1, length(H));
% 
% for i=1:length(H)
%     h = H(i);
%     [tei, uei, itei] = eulero_indietro(f,df, t0, T, y0, h);
%     err(i) = norm(y(tei) -uei, 'inf');
% end
% 
% 
% % figure(1)
% % loglog(H, err, 'k', H, H, '--r')
% 
% err1 = zeros(1, length(H));
% 
% for i=1:length(H)
%     h = H(i);
%     [tei, uei, itei] = crank_nicolson(f,df, t0, T, y0, h);
%     err1(i) = norm(y(tei) -uei, 'inf');
% end
% 
% 
% figure(1)
% loglog(H, err, 'k', H, H, '--r', H, err1, 'b', H, H.^2, '--g')

%% TdE

% n = 10;
% x =1:n;
% 
% A = diag(3*ones(1, n)) + diag(-2 * ones(1, n-1), 1) + diag(-1 * ones(1, n-1), -1);
% b = A*x';
% 
% [L, U, P] = lu(A);
% 
% y = fwsub(L, P*b);
% x = bksub(U, y);
% 
% B = A * A';
% 
% [L1, U1, P1] = lu(B);
% 
% xb_Ex = B\b;
% 
% yb = fwsub(L, P1 * b);
% xb = bksub(U, b);
% 
% r = b - B*xb_Ex;
% K = cond(B)
% 
% err = K * norm(r) ./ norm(b)

%% TdE

%% Sim2.1

% n= 7;
% A = diag(ones(1, n));
% A(:, n) = ones(n, 1);
% A(n, :) = 1:n;
% 
% [L, U, P] = lu(A);
% 
% b= [1; zeros(n-1, 1)];
% 
% y = fwsub(L, P*b);
% x = bksub(U, y);
% 
% A(n, :) = [1.1, ones(1, n-1)];
% 
% N=[20, 30, 40];
% err = zeros(1, length(N));
% 
% for i=1:length(N)
%     n = N(i);
%     A = diag(ones(1, n));
%     A(:, n) = ones(n, 1);
%     A(n, :) = [1.1, ones(1, n-1)];
%     [L, U, P] = lu(A);
%     b = [1; zeros(n-1, 1)];
%     y = fwsub(L, P*b);
%     x = bksub(U, y);
%     x_ex = [1-(1.1/(1.1+n-3)); -(1.1/(1.1+n-3)) * ones(n-2, 1); (1.1/(1.1+n-3))];
%     err(i) = norm(x-x_ex) / norm(x_ex);
% end
% 
% err

%% Sim2.2
% f =@(t, y) -1./t .*(2*y+t.^2.*y.^2);
% to =1;
% T = 5;
% y0 = 1;
% 
% df=@ (t, y) -1./t .*(2+t.^2.*2.*y);
% 
% h= 1;
% [t_ea, u_ea] = eulero_avanti(f, t0, T, y0, h);
% 
% 
% y_ex =@(t) 1./((t.^2) .*(1+log(t)));
% xx = linspace(t0, T, 1000);
% yy = y_ex(xx);
% 
% 
% figure(1)
% plot(xx, yy, t_ea, u_ea)
% 
% H = [0.1, 0.05, 0.025, 0.00125];
% err = zeros(1, length(H));
% 
% for i =1:length(H)
%     h = H(i);
%     [t_ea, u_ea] = eulero_avanti(f, t0, T, y0, h);
%     err(i) = norm(y_ex(t_ea) - u_ea, 'inf');
% end
% 
% figure(1)
% loglog(H, err, H, H)

%% TdE
% f=@(x) cos(7*x).* exp(-2*x);
% df = @(x) -7.*sin(7*x).* exp(-2*x) -2.*cos(7*x) .*exp(-2*x);
% 
% x0 = -2;
% x1=-4;
% toll = 1e-6;
% nmax = 1000;
% 
% [xvec1, it1] = newton(x1, nmax, toll, f,df);
% [xvec0, it0] = newton(x0, nmax, toll, f,df);
% 
% stimap(xvec0);

% g =@(x) (cos(7*x).^4) .* exp(-2*x);
% dg =@(x) -7.*sin(7*x).* exp(-2*x) .* 4.* cos(7*x).^3 -2.*cos(7*x).^4 .*exp(-2*x);
% 
% 
% [xvec3, it3] = newton(x0, nmax, toll, g,dg);
% 
% stimap(xvec3)

%% TdE
% f =@(x) 3* sin(pi/3 .*x) - exp(-3*x);
% a = 0;
% b = 1;
% n = [8, 16, 32, 64];
% 
% ex = 9./2/pi + (exp(-3)-1)/3;
% ex_values = ex .* ones(1, length(n));
% values = zeros(1, length(n));
% for i =1:length(n)
%     values(i) = trapcomp(a,b, n(i), f);
% end
% 
% err = abs(values- ex_values);
% 
% H = (b-a) ./n;
% figure(1)
% loglog(H, err, H, H.^2)


%% TdE

% f =@(t, y) -y .* (2+ 1/(t+1).^2);
% df =@(t, y) -(2+ 1/(t+1).^2);
% t0 = 0;
% T = 10;
% y0 = 10*exp(1);
% 
% y_ex =@(t) 10 .* exp(1./(t+1) -2.*t);
% 
% xx = linspace(t0, T, 1000);
% yy = y_ex(xx);
% 
% h = 0.5;
% [t_cr, u_cr] = crank_nicolson(f, df, t0, T, y0, h);
% 
% 
% h = 1.0;
% [t_ea, u_ea] = eulero_avanti(f, t0, T, y0, h);
% 
% h1 = 0.1;
% [t_ea1, u_ea1] = eulero_avanti(f, t0, T, y0, h1);
% 
% figure(1)
% plot(t_ea, u_ea,'-ro', t_ea1, u_ea1, '-go', xx, yy, 'k')

%% TdE
% 
% f=@(x) sin(10*pi*x)./ (1+10.*x.^2);
% 
% a=-3;
% b= 3;
% 
% n = [6, 8, 10, 14];
% x_signed = 1- pi/10;
% 
% x_value = zeros(1, length(n));
% for i=1:length(n)
%     x_dis = linspace(a, b, n(i)+1);
%     P = polyfit(x_dis, f(x_dis), n(i));
%     x_value(i) = polyval(P, x_signed);
% end
% 
% x_value
% 
% err = abs(f(x_signed) - x_value)

n= 7;
A = diag(ones(1, n));
A(:, n) = ones(n, 1);
A(n, :) = 1:n;

[L, U, P] = lu(A);

b= [1; zeros(n-1, 1)];

y = fwsub(L, P*b);
x = bksub(U, y);

A(n, :) = [1.1, ones(1, n-1)];

N=[20, 30, 40];
err = zeros(1, length(N));

for i=1:length(N)
    n = N(i);
    A = diag(ones(1, n));
    A(:, n) = ones(n, 1);
    A(n, :) = [1.1, ones(1, n-1)];
    [L, U, P] = lu(A);
    b = [1; zeros(n-1, 1)];
    y = fwsub(L, P*b);
    x = bksub(U, y);
    x_ex = [1-(1.1/(1.1+n-3)); -(1.1/(1.1+n-3)) * ones(n-2, 1); (1.1/(1.1+n-3))];
    err(i) = norm(x-x_ex) / norm(x_ex);
end

err

