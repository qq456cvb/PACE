function [R_t_2,R_t_3,Reconst,T,iter]=NordbergTFTPoseEstimation(Corresp,CalM)
%NORDBERGTFTPOSEESTIMATION Pose estimation of 3 views from corresponding
% triplets of points using the minimal TriFocal Tensor parameterization
% by K. Nordberg.
%
%  An initial trifocal tensor is computed linearly from the trilinearities
%  using the triplets of correspondences. Then the error is minimized using
%  Gauss-Helmert model to impose the minimal constraints of the K. Nordberg
%  parameterization. After the optimization the essential matrices are
%  computed from the tensor and the orientations are extracted by SVD.
%
%  Input arguments:
%  Corresp  - 6xN matrix containing in each column, the 3 projections of
%             the same space point onto the 3 images.
%  CalM     - 9x3 matrix containing the M calibration 3x3 matrices for 
%             each camera concatenated.
%
%  Output arguments: 
%  R_t_2    - 3x4 matrix containing the rotation matrix and translation 
%             vector [R2,t2] for the second camera.
%  R_t_3    - 3x4 matrix containing the rotation matrix and translation 
%             vector [R3,t3] for the third camera.
%  Reconst  - 3xN matrix containing the 3D reconstruction of the
%             correspondences.
%  T        - 3x3x3 array containing the trifocal tensor associated to 
%             this triplet of cameras.
%  iter     - number of iterations needed in GH algorithm to reach minimum 
%

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


% Normalization of the data
[x1,Normal1]=Normalize2Ddata(Corresp(1:2,:));
[x2,Normal2]=Normalize2Ddata(Corresp(3:4,:));
[x3,Normal3]=Normalize2Ddata(Corresp(5:6,:));

% Model to estimate T: linear equations
[T,P1,P2,P3]=linearTFT(x1,x2,x3);

% apply projective transformation so matrix B ( P3=[B|b] ) has full rank
H=eye(4);
if rank(P3(:,1:3))<3
    H(4,1:3)=null(P3(:,1:3))';
elseif rank(P2(:,1:3))<3
    H(4,1:3)=null(P2(:,1:3))';
end
P1=P1*H; P2=P2*H; P3=P3*H;
    
% Compute K. Nordberg param
A=P2(:,1:3); a=P2(:,4); r=A\a;
B=P3(:,1:3); b=P3(:,4); s=B\b;

U=[r, crossM(r)^2*s, crossM(r)*s];      U=U*(U'*U)^(-1/2); U=sign(det(U))*U;
V=[a, crossM(a)*A*s, crossM(a)^2*A*s];  V=V*(V'*V)^(-1/2); V=sign(det(V))*V;
W=[b, crossM(b)*B*r, crossM(b)^2*B*r];  W=W*(W'*W)^(-1/2); W=sign(det(W))*W;

% good representation of U V W
[~,~,v]=svd(U-eye(3));  vec_u=v(:,3);
o_u=atan2(vec_u'*[U(3,2)-U(2,3);U(1,3)-U(3,1);U(2,1)-U(1,2)]/2, (trace(U)-1)/2);
[~,~,v]=svd(V-eye(3));  vec_v=v(:,3);
o_v=atan2(vec_v'*[V(3,2)-V(2,3);V(1,3)-V(3,1);V(2,1)-V(1,2)]/2, (trace(V)-1)/2);
[~,~,v]=svd(W-eye(3));  vec_w=v(:,3);
o_w=atan2(vec_w'*[W(3,2)-W(2,3);W(1,3)-W(3,1);W(2,1)-W(1,2)]/2, (trace(W)-1)/2);

% sparce tensor
Ts=transf_t(T,U,V,W);
paramT=Ts([1,7,10,12,16,19:22,25])';
paramT=paramT/norm(paramT);

% compute 3D estimated points to have initial estimated reprojected image
% points
points3D=triangulation3D({P1,P2,P3},[x1;x2;x3]);
p1_est=P1*points3D; p1_est=p1_est(1:2,:)./repmat(p1_est(3,:),2,1);
p2_est=P2*points3D; p2_est=p2_est(1:2,:)./repmat(p2_est(3,:),2,1);
p3_est=P3*points3D; p3_est=p3_est(1:2,:)./repmat(p3_est(3,:),2,1);

% minimize error with Gauss-Helmert
N=size(x1,2);
param0=[vec_u*o_u; vec_v*o_v; vec_w*o_w;paramT];
obs=reshape([x1(1:2,:);x2(1:2,:);x3(1:2,:)],6*N,1);
obs_est=reshape([p1_est;p2_est;p3_est],6*N,1);
[~,param,~,iter]=Gauss_Helmert(@constrGH,obs_est,param0,zeros(0,1),obs,eye(6*N));

% recover orthogonal matrices from optimized parameters
o_u=norm(param(1:3)); vec_u=param(1:3)/o_u;
o_v=norm(param(4:6)); vec_v=param(4:6)/o_v;
o_w=norm(param(7:9)); vec_w=param(7:9)/o_w;
U=eye(3)+sin(o_u)*crossM(vec_u)+(1-cos(o_u))*crossM(vec_u)^2;
V=eye(3)+sin(o_v)*crossM(vec_v)+(1-cos(o_v))*crossM(vec_v)^2;
W=eye(3)+sin(o_w)*crossM(vec_w)+(1-cos(o_w))*crossM(vec_w)^2;

% sparse tensor from optimized parameters
Ts=zeros(3,3,3);
Ts([1,7,10,12,16,19:22,25])=param(10:19);

% trifocal tensor
T=transf_t(Ts,U',V',W');

% denormalization
T= transform_TFT(T,Normal1,Normal2,Normal3,1);

% Find orientation using calibration and TFT
[R_t_2,R_t_3]=R_t_from_TFT(T,CalM,Corresp);

% Find 3D points by triangulation
Reconst=triangulation3D({CalM(1:3,:)*eye(3,4),CalM(4:6,:)*R_t_2,CalM(7:9,:)*R_t_3},Corresp);
Reconst=Reconst(1:3,:)./repmat(Reconst(4,:),3,1);

end

%%% function with GH constraints and parameters for Nordberg's
%%% parameterization
function [f,g,A,B,C,D]=constrGH(obs,x,~)

% orthogonal matrices
o_u=norm(x(1:3)); vec_u=x(1:3)/o_u;
o_v=norm(x(4:6)); vec_v=x(4:6)/o_v;
o_w=norm(x(7:9)); vec_w=x(7:9)/o_w;
U=eye(3)+sin(o_u)*crossM(vec_u)+(1-cos(o_u))*crossM(vec_u)^2;
V=eye(3)+sin(o_v)*crossM(vec_v)+(1-cos(o_v))*crossM(vec_v)^2;
W=eye(3)+sin(o_w)*crossM(vec_w)+(1-cos(o_w))*crossM(vec_w)^2;

% sparse tensor
paramT=x(10:19);
Ts=zeros(3,3,3);
param_ind=[1,7,10,12,16,19:22,25];
Ts(param_ind)=paramT;

% original tensor
T=transf_t(Ts,U',V',W');

% observations
obs=reshape(obs,6,[]);
N=size(obs,2);

f=zeros(4*N,1);     % constraints for tensor and observations (trilinearities)
Ap=zeros(4*N,27);   % jacobian of f w.r.t. the tensor T
B=zeros(4*N,6*N);   % jacobian of f w.r.t. the observations
for i=1:N
    % points in the three images for correspondence i
    x1=obs(1:2,i); x2=obs(3:4,i); x3=obs(5:6,i);
    
    % 4 trilinearities
    ind2=4*(i-1);
    S2=[0 -1; -1 0; x2(2) x2(1)];
    S3=[0 -1; -1 0; x3(2) x3(1)];
    f(ind2+1:ind2+4)=reshape( S2'*(x1(1)*T(:,:,1)+x1(2)*T(:,:,2)+T(:,:,3))*S3,4,1);
    
    % Jacobians for the trilinearities
    Ap(ind2+1:ind2+4,:)=kron(S3,S2)'*kron([x1;1]',eye(9));
    B(ind2+1:ind2+4,6*(i-1)+1)=reshape(S2'*T(:,:,1)*S3,4,1);
    B(ind2+1:ind2+4,6*(i-1)+2)=reshape(S2'*T(:,:,2)*S3,4,1);
    B(ind2+1:ind2+4,6*(i-1)+(3:4))=kron(S3'*reshape(T(3,:,:),3,3)*[x1;1],[0,1;1,0]);
    B(ind2+1:ind2+4,6*(i-1)+(5:6))=kron([0,1;1,0],S2'*reshape(T(:,3,:),3,3)*[x1;1]);
end


% Jacobian of T=Ts(U,V,W) w.r.t. the minimal parameterization
J=zeros(27,19);
% derivatives of the parameterization w.r.t. the sparse tensor
for i=1:10
    e=zeros(3,3,3); e(param_ind(i))=1;
    J(:,i+9)=reshape(transf_t(e,U',V',W'),27,1);
end
% derivatives of the parameterization w.r.t. the orthogonal matrices
dU=zeros(3,3,3); dV=zeros(3,3,3); dW=zeros(3,3,3);
e=eye(3);
for i=1:3
    dU(:,:,i)=-vec_u(i)*sin(o_u)*eye(3)+ vec_u(i)*cos(o_u)*crossM(vec_u)+...
        sin(o_u)*(1/o_u)*(crossM(e(:,i))- vec_u(i)*crossM(vec_u))+...
        vec_u(i)*sin(o_u)*(vec_u*vec_u')+...
        (1-cos(o_u))*(1/o_u)*(vec_u*e(i,:)+e(:,i)*vec_u'-2*vec_u(i)*(vec_u*vec_u'));
    
    dV(:,:,i)=-vec_v(i)*sin(o_v)*eye(3)+ vec_v(i)*cos(o_v)*crossM(vec_v)+...
        sin(o_v)*(1/o_v)*(crossM(e(:,i))- vec_v(i)*crossM(vec_v))+...
        vec_v(i)*sin(o_v)*(vec_v*vec_v')+...
        (1-cos(o_v))*(1/o_v)*(vec_v*e(i,:)+e(:,i)*vec_v'-2*vec_v(i)*(vec_v*vec_v'));
    
    dW(:,:,i)=-vec_w(i)*sin(o_w)*eye(3)+ vec_w(i)*cos(o_w)*crossM(vec_w)+...
        sin(o_w)*(1/o_w)*(crossM(e(:,i))- vec_w(i)*crossM(vec_w))+...
        vec_w(i)*sin(o_w)*(vec_w*vec_w')+...
        (1-cos(o_w))*(1/o_w)*(vec_w*e(i,:)+e(:,i)*vec_w'-2*vec_w(i)*(vec_w*vec_w'));
end
for i=1:3
    J(:,i)=reshape(transf_t(Ts,dU(:,:,i)',V',W'),27,1);
    J(:,i+3)=reshape(transf_t(Ts,U',dV(:,:,i)',W'),27,1);
    J(:,i+6)=reshape(transf_t(Ts,U',V',dW(:,:,i)'),27,1);
end


A=Ap*J; % Jacobian of f w.r.t. the minimal parameterization 

g=sum(paramT.^2)-1;     % constraints on the minimal parameterization
C=zeros(1,19);          % jacobian of g w.r.t. the minimal parameterization
C(1,10:19)=2*paramT';
D=zeros(1,0);

end


%%% Operation on a tensor by 3 matrices
function T=transf_t(T0,U,V,W)
T=zeros(3,3,3);
for i=1:3
    T(:,:,i)=V' *(U(1,i)*T0(:,:,1) +U(2,i)*T0(:,:,2) +U(3,i)*T0(:,:,3))* W;
end
end


