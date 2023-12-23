function [R_t_2,R_t_3,Reconst,T,iter]=LinearFPoseEstimation(Corresp,CalM)
%LINEARFPOSEESTIMATION Pose estimation of 3 views from corresponding
% triplets of points using the linear estimation of fundamental matrices. 
% 
%  The fundamental matrices are computed with an algebraic minimization of
%  the epipolar equations for two of the three possible pairs of views. The
%  essential matrices are computed with the calibration  matrices. The 
%  orientations are extracted by SVD.
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


N=size(Corresp,2);
K1=CalM(1:3,:); K2=CalM(4:6,:); K3=CalM(7:9,:);

% Normalization of the data
[x1,Normal1]=Normalize2Ddata(Corresp(1:2,:));
[x2,Normal2]=Normalize2Ddata(Corresp(3:4,:));
[x3,Normal3]=Normalize2Ddata(Corresp(5:6,:));

% the 3 fundamental matrices
F21=linearF(x1,x2);
F31=linearF(x1,x3);

% Undo normalization
F21=Normal2.'*F21*Normal1; 
F31=Normal3.'*F31*Normal1;

% Find orientation using calibration and F matrices
[R2,t2]=recover_R_t(K1,K2,F21,Corresp(1:2,:),Corresp(3:4,:));
[R3,t3]=recover_R_t(K1,K3,F31,Corresp(1:2,:),Corresp(5:6,:));

% Find the norm of t31 using the image points and reconstruction from
% images 1 and 2
u3=K3*t3;
X=triangulation3D({K1*eye(3,4),K2*[R2,t2]},Corresp(1:4,:));
X=X(1:3,:)./repmat(X(4,:),3,1);
X3=K3*R3*X;
lam=-sum(dot(cross([Corresp(5:6,:);ones(1,N)],X3,1),cross([Corresp(5:6,:);ones(1,N)],repmat(u3,1,N)),1))/...
    sum(sum(cross([Corresp(5:6,:);ones(1,N)],repmat(u3,1,N)).^2));
t3=lam*t3;

R_t_2=[R2,t2]; R_t_3=[R3,t3];

% Find 3D points by triangulation
Reconst=triangulation3D({K1*eye(3,4),K2*R_t_2,K3*R_t_3},Corresp);
Reconst=Reconst(1:3,:)./repmat(Reconst(4,:),3,1);
iter=0;
T=TFT_from_P(K1*eye(3,4),K2*R_t_2,K3*R_t_3);

end


%%% Extracts rotation and translation from fundamental matrix
function [R_f,t_f]=recover_R_t(K1,K2,F21,x1,x2)

E21=K2.'*F21*K1;
W=[0 -1 0; 1 0 0; 0 0 1];
[U,~,V]=svd(E21);
R=U*W*V.';  Rp=U*W.'*V.';
R=R*sign(det(R)); Rp=Rp*sign(det(Rp));
t=U(:,3);

%from the 4 possible solutions find the correct one using the image points
num_points_seen=0;
for k=1:4
    if k==2 || k==4
        t=-t;
    elseif k==3
        R=Rp;
    end
    X1=triangulation3D({[K1 [0;0;0]],K2*[R,t]},[x1;x2]); X1=X1./repmat(X1(4,:),4,1);
    X2=[R t]*X1;
    if sum(sign(X1(3,:))+sign(X2(3,:)))>=num_points_seen
        R_f=R; t_f=t;
        num_points_seen=sum(sign(X1(3,:))+sign(X2(3,:)));
    end
end

end

