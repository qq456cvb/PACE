function [R_t_2,R_t_3,Reconst,T,iter]=ResslTFTPoseEstimation(Corresp,CalM)
%RESSLTFTPOSEESTIMATION Pose estimation of 3 views from corresponding
% triplets of points using the minimal TriFocal Tensor parameterization
% by C. Ressl.
%
%  An initial trifocal tensor is computed linearly from the trilinearities
%  using the triplets of correspondences. Then the error is minimized using
%  Gauss-Helmert model to impose the minimal constraints of C. Ressl TFT
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

% Compute Ressl parameters
e21=P2(:,4);
[~,Ind]=max(abs(e21)); e21=e21/e21(Ind);
e31=P3(:,4);           e31=e31/norm(e31);

S=[T(Ind,:,1).',T(Ind,:,2).',T(Ind,:,3).'];
aux=norm(S(:));     S=S./aux;
T=T/aux;

Ind2=1:3; Ind2=Ind2(Ind2~=Ind);
mn=[e31.'*(T(:,:,1).'-S(:,1)*e21.');...
    e31.'*(T(:,:,2).'-S(:,2)*e21.');...
    e31.'*(T(:,:,3).'-S(:,3)*e21.')];
mn=mn(:,Ind2);

% compute 3D estimated points to have initial estimated reprojected image
% points
points3D=triangulation3D({P1,P2,P3},[x1;x2;x3]);
p1_est=P1*points3D; p1_est=p1_est(1:2,:)./repmat(p1_est(3,:),2,1);
p2_est=P2*points3D; p2_est=p2_est(1:2,:)./repmat(p2_est(3,:),2,1);
p3_est=P3*points3D; p3_est=p3_est(1:2,:)./repmat(p3_est(3,:),2,1);

% minimize error with Gauss-Helmert
N=size(x1,2);
p=[S(:);e21(Ind2);mn(:);e31];
x=reshape([x1(1:2,:);x2(1:2,:);x3(1:2,:)],6*N,1);
x_est=reshape([p1_est;p2_est;p3_est],6*N,1);
y=zeros(0,1);
func=@(x1,x2,x3)constraintsGH(x1,x2,x3,Ind);
[~,p_opt,~,iter]=Gauss_Helmert(func,x_est,p,y,x,eye(6*N));

% recover parameters
S=reshape(p_opt(1:9),3,3);
e21=ones(3,1); e21(Ind2)=p_opt(10:11);
mn=zeros(3);   mn(:,Ind2)=reshape(p_opt(12:17),3,2);
e31=p_opt(18:20);
% estimated tensor
T(:,:,1)=(S(:,1)*e21.'+e31*mn(1,:)).';
T(:,:,2)=(S(:,2)*e21.'+e31*mn(2,:)).';
T(:,:,3)=(S(:,3)*e21.'+e31*mn(3,:)).';
% denormalization
T= transform_TFT(T,Normal1,Normal2,Normal3,1);

% Find orientation using calibration and TFT
[R_t_2,R_t_3]=R_t_from_TFT(T,CalM,Corresp);

% Find 3D points by triangulation
Reconst=triangulation3D({CalM(1:3,:)*eye(3,4),CalM(4:6,:)*R_t_2,CalM(7:9,:)*R_t_3},Corresp);
Reconst=Reconst(1:3,:)./repmat(Reconst(4,:),3,1);

end


%%% function with GH constraints and parameters for Ressl's
%%% parameterization
function [f,g,A,B,C,D]=constraintsGH(x,p,~,Ind)

Ind2=1:3; Ind2=Ind2(Ind2~=Ind);
% recover Ressl's paremeters
S=reshape(p(1:9),3,3);
e21=ones(3,1);  e21(Ind2)=p(10:11);  e31=p(18:20);
mn=zeros(3);    mn(:,Ind2)=reshape(p(12:17),3,2);
% tensor
T1=(S(:,1)*e21.'+e31*mn(1,:)).';
T2=(S(:,2)*e21.'+e31*mn(2,:)).';
T3=(S(:,3)*e21.'+e31*mn(3,:)).';

% other slices of the TFT
J3=[T1(3,:);T2(3,:);T3(3,:)];
K3=[T1(:,3),T2(:,3),T3(:,3)];

N=size(x,1)/6;

% constraints evaluated in (p)
g=[sum(e31.^2)-1;sum(S(:).^2)-1];


% g jacobian w.r.t. p evaluated in p
C=zeros(2,20);
C(1,18:20)=2*e31.';
C(2,1:9)=2*S(:).';


f=zeros(4*N,1);     % constraints for tensor and observations (trilinearities)
Ap=zeros(4*N,27);   % jacobian of f w.r.t. the tensor T
B=zeros(4*N,6*N);   % jacobian of f w.r.t. the observations
for i=1:N
    
    % points in the three images for correspondance i
    ind=6*(i-1);
    x1=x(ind+1:ind+2);
    x2=x(ind+3:ind+4);
    x3=x(ind+5:ind+6);
    
    % 4 trilinearities
    ind2=4*(i-1);
    S2=[0 -1; -1 0; x2(2) x2(1)];
    S3=[0 -1; -1 0; x3(2) x3(1)];
    f(ind2+1:ind2+4)=reshape( S2.'*(x1(1)*T1+x1(2)*T2+T3)*S3,4,1);
    
    % Jacobians for the trilinearities
    Ap(ind2+1:ind2+4,:)=kron(S3,S2).'*kron([x1;1].',eye(9));
    B(ind2+1:ind2+4,ind+1)=reshape(S2.'*T1*S3,4,1);
    B(ind2+1:ind2+4,ind+2)=reshape(S2.'*T2*S3,4,1);
    B(ind2+1:ind2+4,ind+3:ind+4)=kron(S3.'*J3.'*[x1;1],[0,1;1,0]);
    B(ind2+1:ind2+4,ind+5:ind+6)=kron([0,1;1,0],S2.'*K3*[x1;1]);
end

% jacobian for parametrization t=F(p) w.r.t. p evaluated in p
D=zeros(27,20);
D(:,1:9)=kron(eye(3),kron(eye(3),e21));
aux=zeros(3,2); aux(Ind2,:)=eye(2);
D(:,10:11)=[kron(S(:,1),aux);kron(S(:,2),aux);kron(S(:,3),aux)];
D(:,12:14)=kron(eye(3),kron(e31,aux(:,1)));
D(:,15:17)=kron(eye(3),kron(e31,aux(:,2)));
D(:,18:20)=[kron(eye(3),mn(1,:).');kron(eye(3),mn(2,:).');kron(eye(3),mn(3,:).')];

% jacobian of f w.r.t. the minimal parameterization
A=Ap*D;

D=zeros(2,0);

end
    



