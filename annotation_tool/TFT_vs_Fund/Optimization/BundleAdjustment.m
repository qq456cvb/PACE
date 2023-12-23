function [R_t,Reconst,iter,repr_err]=BundleAdjustment(CalM,R_t_0,Corresp,Reconst0)
%BUNDLE ADJUSTMENT Bundle adjustment optimization.
%
%  Bundle Adjustment for the pose estimation of M cameras and N 3D points.
%  The reprojection error of the N points to the M cameras is minimized over
%  the possible positions of the space points and orientations of the
%  cameras. The points do not need to be seen in all cameras but at least in
%  two. An initial guess for the orientations of the cameras is needed while
%  an initial triangulation of the space points is optional. The
%  oprimization is carried out by the Levenberg-Marquardt algorithm. The
%  rotations are parametrized by three angles each.
%
%  Input arguments:
%  CalM     - 3Mx3 matrix containing the M calibration 3x3 matrices for 
%             each camera concatenated.
%  R_t_0    - 3Mx4 matrix of M 3x4 matrices concatenated containing a first
%             pose estimation [R t] for each camera.
%  Corresp  - 2MxN matrix containing in each column, the M projections of
%             the same space point onto the M images. If point n is not
%             seen in image m, then Corresp(2*m-1:2*m,n)=[NaN;NaN].
%  Reconst0 - 3xN matrix containing an initial estimation of the N 3D
%             points. If not provided, they will be estimated.
%
%  Output arguments:
%  R_t      - 3Mx4 matrix of M 3x4 matrices concatenated with the final 
%             rotation and translation of each camera, [R t].
%  Reconst  - 3xN matrix containing the final estimation of the N 3D points.
%  iter     - number of iterations needed in L-M algorithm to reach
%             the minimum.
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


M=size(Corresp,1)/2;    % Number of total images
N=size(Corresp,2);      % Number of total 3D points to recover

% Normalization of image points
for j=1:M
    [new_Corr,Normal]=Normalize2Ddata(Corresp(2*j-1:2*j,:));
    Corresp(2*j-1:2*j,:)=new_Corr;
    CalM(3*j-2:3*j,:)=Normal*CalM(3*j-2:3*j,:);
end

% If not provided, we compute a first triangulation of the 3D points
if nargin<4
    Reconst0=zeros(3,N);
    for i=1:N
        points=[];
        cameras={};
        aux=0;
        for j=1:M
            if isnan(Corresp(2*j-1,i))
                continue;
            end
            aux=aux+1;
            points=[points; Corresp(2*j-1:2*j,i)];
            cameras={cameras{1:(aux-1)},...
                CalM(3*j-2:3*j,:)*R_t_0(3*j-2:3*j,:)};
        end
        X = triangulation3D(cameras,points);
        Reconst0(:,i)=X(1:3)/X(4);
    end
end

% Change coordinates so that the first pose is [ Id 0 ]
change_coord=R_t_0(1:3,:);
R_t_0(1:3,:)=eye(3,4);
for j=2:M
    R_t_0(3*j-2:3*j,4)  = R_t_0(3*j-2:3*j,4) -R_t_0(3*j-2:3*j,1:3)*change_coord(:,1:3).'*change_coord(:,4);
    R_t_0(3*j-2:3*j,1:3)= R_t_0(3*j-2:3*j,1:3)*change_coord(:,1:3).';
end
Reconst0=change_coord(:,1:3)*Reconst0 + repmat(change_coord(:,4),1,N);

% Compute rotations parametrization by angles and translations
angles0=zeros(3,M);
translations0=zeros(3,M);
for j=1:M
    angles0(1,j)=-atan2( R_t_0(3*j-1,3),R_t_0(3*j,3));
    angles0(2,j)=-atan2(-R_t_0(3*j-2,3),norm(R_t_0(3*j-1:3*j,3)));
    angles0(3,j)=-atan2( R_t_0(3*j-2,2),R_t_0(3*j-2,1));
    translations0(:,j)=R_t_0(3*j-2:3*j,4);
end

% Optimization using Levenberg - Marquardt
func=@(x)bundleadjustment_LM(x,Corresp,CalM);
variables0=reshape([angles0(:,2:M), translations0(:,2:M), Reconst0],[],1);
options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt','Jacobian','on','Display','off');
[variables,~,~,~,output]=lsqnonlin(func,variables0,[],[],options);

% Final reprojection error
repr_err=norm(func(variables));

% Total iterations in the optimization
iter=output.iterations;

% Recover final reconstruction and orientations & fix scale
variables=reshape(variables,3,N+2*(M-1));
angles=variables(:,1:(M-1));
translations=variables(:,M:2*(M-1)); scale=1/norm(translations(:,1));
R_t=zeros(3*M,4); R_t(1:3,:)=eye(3,4);
for j=1:(M-1)
    Rx=[1 0 0; 0 cos(angles(1,j)) -sin(angles(1,j)); 0 sin(angles(1,j)) cos(angles(1,j))];
    Ry=[cos(angles(2,j)) 0 sin(angles(2,j)); 0 1 0; -sin(angles(2,j)) 0 cos(angles(2,j))];
    Rz=[cos(angles(3,j)) -sin(angles(3,j)) 0; sin(angles(3,j)) cos(angles(3,j)) 0; 0 0 1];
    R_t(3*j+(1:3),:)=[Rx*Ry*Rz, scale*translations(:,j)];
end
Reconst=scale*variables(:,2*(M-1)+1:2*(M-1)+N);
end



%%% function for the Least Squares minimisation with Levenberg-Marquardt
function [f,J]=bundleadjustment_LM(variables,Corresp,CalM)
% the first camera matrix is supposed to be K1 * [ Id 0 ]

N=size(Corresp,2);    % number of total 3d points to recover
M=size(Corresp,1)/2;  % number of total images

% recover rotations & translations from variables
variables=reshape(variables,3,N+2*(M-1));
angles=variables(:,1:(M-1));
translations=variables(:,M:2*(M-1));
space_points=variables(:,2*(M-1)+1:2*(M-1)+N);

Rot=repmat(eye(3*M,3),1,3); dRot=zeros(3*M,9); 
Pcam=zeros(3*M,4); 
Pcam(1:3,:)=CalM(1:3,:)*eye(3,4);
for j=1:(M-1)
    % camera number j+1
    Rx=[1 0 0; 0 cos(angles(1,j)) -sin(angles(1,j)); 0 sin(angles(1,j)) cos(angles(1,j))];
    Ry=[cos(angles(2,j)) 0 sin(angles(2,j)); 0 1 0; -sin(angles(2,j)) 0 cos(angles(2,j))];
    Rz=[cos(angles(3,j)) -sin(angles(3,j)) 0; sin(angles(3,j)) cos(angles(3,j)) 0; 0 0 1];
    Rot(3*j+(1:3),:)=[Rx,Ry,Rz];
    
    Dx=zeros(3); Dx(2:3,2:3)=[-sin(angles(1,j)) -cos(angles(1,j)); cos(angles(1,j)) -sin(angles(1,j))];
    Dy=zeros(3); Dy([1 3],[1 3])=[-sin(angles(2,j)) cos(angles(2,j)); -cos(angles(2,j)) -sin(angles(2,j))];
    Dz=zeros(3); Dz(1:2,1:2)=[-sin(angles(3,j)) -cos(angles(3,j)); cos(angles(3,j)) -sin(angles(3,j))];
    dRot(3*j+(1:3),:)=[Dx,Dy,Dz];
    
    Pcam(3*j+(1:3),:)=CalM(3*j+(1:3),:)*[Rx*Ry*Rz, translations(:,j)];
end


f=zeros(N*M*2,1);
J=zeros(N*M*2,3*(2*(M-1)+N));
for i=1:N
    % 3D point corresponding to correspondence i
    Point=space_points(:,i);
    for j=1:M
        if isnan(Corresp(2*j-1,i))
            continue;
        end
        
        % Camera j
        ind_cam=3*j-2:3*j;
        P=Pcam(ind_cam,:);       K=CalM(ind_cam,:);
        dR_x=dRot(ind_cam,1:3);  R_x=Rot(ind_cam,1:3);
        dR_y=dRot(ind_cam,4:6);  R_y=Rot(ind_cam,4:6);
        dR_z=dRot(ind_cam,7:9);  R_z=Rot(ind_cam,7:9);
        
        % point in image j for correspondence i
        point=Corresp(2*j-1:2*j,i);

        ind=2*M*(i-1)+2*(j-1);
        % f: distance from p to projection P_j*P
        [aux,dgamma]=Gamma(P*[Point;1]);
        [aux,~,dydist]=Dist(point,aux);
        f(ind+1:ind+2)=aux;

        % Jacobians for f
        Jac=zeros(3,3*(2*(M-1)+N));
        
        % respect 3d point
        Jac(:,6*(M-1)+3*(i-1)+(1:3))=P(:,1:3);
        
        if j>1
            %respect translation of camera j
            Jac(:,3*(M-1)+3*(j-2)+(1:3))=K;

            %respect rotation (angles)
            Jac(:,3*(j-2)+(1:3))=[K*(dR_x*R_y*R_z)*Point,...
                    K*(R_x*dR_y*R_z)*Point, K*(R_x*R_y*dR_z)*Point];
        end
        
        J(ind+1:ind+2,:)=dydist*dgamma*Jac;
    end
end

end

function [f,dfx,dfy]=Dist(x,y)
f=x-y;
dfx=eye(2);
dfy=-eye(2);
end

function [f,df]=Gamma(v)
f=v(1:2)/v(3);
df=[(1/v(3))*eye(2), [-v(1)/(v(3)^2);-v(2)/(v(3)^2)]];
end

