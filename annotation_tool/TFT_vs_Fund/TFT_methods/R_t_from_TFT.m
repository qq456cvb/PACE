function [R_t_2,R_t_3]=R_t_from_TFT(T,CalM,Corresp)
%R_T_FROM_TFT Pose estimation of 3 views from the associated TriFocal
% Tensor and the calibration matrices. 
%
%  Form the TFT, the epipoles are computed and then the essential matrices.
%  The orientations are extracted by SVD.
%
%  Input arguments:
%  T        - 3x3x3 array containing the trifocal tensor associated to this
%             triplet of cameras
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

% 'remove' calibration from the tensor
T= transform_TFT(T,K1,K2,K3,1);

% epipoles and essential matrix
[~,~,V]=svd(T(:,:,1)); v1=V(:,end);
[~,~,V]=svd(T(:,:,2)); v2=V(:,end);
[~,~,V]=svd(T(:,:,3)); v3=V(:,end);
[~,~,V]=svd([v1 v2 v3].'); epi31=V(:,end)*sign(V(end));

[~,~,V]=svd(T(:,:,1).'); v1=V(:,end);
[~,~,V]=svd(T(:,:,2).'); v2=V(:,end);
[~,~,V]=svd(T(:,:,3).'); v3=V(:,end);
[~,~,V]=svd([v1 v2 v3].'); epi21=V(:,end)*sign(V(end));

E21=crossM(epi21)*[T(:,:,1)*epi31 T(:,:,2)*epi31 T(:,:,3)*epi31];
E31=-crossM(epi31)*[T(:,:,1).'*epi21 T(:,:,2).'*epi21 T(:,:,3).'*epi21];

% Find R2 and t2 from E21 
[R2,t2]=recover_R_t(E21,K1,K2,Corresp(1:2,:),Corresp(3:4,:));

% Find R3 and t3 from E31 
[R3,t3]=recover_R_t(E31,K1,K3,Corresp(1:2,:),Corresp(5:6,:));

% Find the norm of t3 using the image points and reconstruction from
% images 1 and 2
u3=K3*t3;
X=triangulation3D({K1*eye(3,4),K2*[R2,t2]},Corresp(1:4,:));
X=X(1:3,:)./repmat(X(4,:),3,1);
X3=K3*R3*X;
lam=-sum(dot(cross([Corresp(5:6,:);ones(1,N)],X3,1),cross([Corresp(5:6,:);ones(1,N)],repmat(u3,1,N)),1))/...
    sum(sum(cross([Corresp(5:6,:);ones(1,N)],repmat(u3,1,N)).^2));
t3=lam*t3;

R_t_2=[R2,t2]; R_t_3=[R3,t3];

end


%%% Extracts rotation and translation from essential matrix
function [R_f,t_f]=recover_R_t(E21,K1,K2,x1,x2)

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
    X1=triangulation3D({K1*eye(3,4),K2*[R,t]},[x1;x2]); X1=X1./repmat(X1(4,:),4,1);
    X2=[R t]*X1;
    if sum(sign(X1(3,:))+sign(X2(3,:)))>=num_points_seen
        R_f=R; t_f=t;
        num_points_seen=sum(sign(X1(3,:))+sign(X2(3,:)));
    end
end

end



