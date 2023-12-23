function [x_opt,t_opt,y_opt,iter]=Gauss_Helmert(func,x0,t0,y0,x,P)
%GAUSS_HELMERT Gauss - Helmert model (general case of least squares adjustment)
%
%  Input parameters:
%  func    - matlab function containing the condition equations f, with
%            jacobians A and B, and constraints g, with jacobians C and D
%            [f,g,A,B,C,D]=func(x,t,y,auxiliar_P)
%  x0      - initial point for the estimation of true image correspondances
%            fitting t0
%  t0      - initial estimation for parameters
%  y0      - initial estimation for additional unknown parameters
%  x       - observations
%  P       - weight matrix for observations
%
%  Output arguments:
%  x_opt   - optimum for the image correspondences
%  t_opt   - optimum parameters
%  y_opt   - optimum additional unknown parameters
%  iter    - total iterations needed to reach minimum

% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>
% All rights reserved.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


it_max=400;
tol=1e-6;

xi=x0; yi=y0; ti=t0;
u=size(t0,1);
s=size(y0,1);

v0=x0-x;
objFunc=sum(v0.'*P*v0);
factor=1;

for it=1:it_max
    [f,g,A,B,C,D]=func(xi,ti,yi);
    c2=size(C,1);
    W=B*pinv(P)*B.';
    if any(isnan(W(:))) || any(isinf(W(:)))
        break;
    end
    % temporary change from W=pinv(W) to W=inv(W)
    W=pinv(W+(1e-12)*eye(size(W,1)));   W=W+(1e-12)*eye(size(W,1));
    w=-f-B*(x-xi);
    M=[A.'*W*A, zeros(u,s), C.';...
              zeros(s,u+s), D.';...
            C, D, zeros(c2,c2)];
    b=[A.'*W*w; zeros(s,1); -g];
    if any(isnan(M(:))) || any(isinf(M(:)))
        break;
    end
    % temporary change from aux=pinv(M)*b; to a=M\b;
    aux=pinv(M+(1e-12)*eye(size(M,1)))*b;
    dt=aux(1:u,:); dy=aux(u+1:u+s,:);
    v=-inv(P)*B.'*(W*(A*dt-w));
    
    if norm(dt)< tol && norm(dy) < tol && norm(xi-x-v) <tol
        break;
    end
    %fprintf('it=%d repr err=%f\n',it,v.'*P*v);
    if sum(v.'*P*v) > objFunc*factor
        break;
    else
        objFunc=sum(v.'*P*v);
    end
    xi=x+v; ti=ti+dt; yi=yi+dy;
end
iter=it;
x_opt=xi; y_opt=yi; t_opt=ti;

end