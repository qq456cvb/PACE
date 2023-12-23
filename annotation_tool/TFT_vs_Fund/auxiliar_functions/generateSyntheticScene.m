function [CalM,R_t,Corresp,points3D]=generateSyntheticScene(N,noise,seed,focalL,angle)
%GENERATESYNTHETICSCENE synthetic data generation
%
%  Generates a synthetic scene composed of three cameras, N 3D points and
%  their projections onto the three images.
%
%  Input arguments:
%  N        - number of random 3D points to generate
%  noise    - sigma in pixels for the gaussian noise in image points
%  seed     - seed for random generation of 3D points and noise
%  focalL   - focal length. Changing this parameter from 50mm, will change 
%             also the coordinates of the camera centers ans ppal axis
%  angle    - angle between the three centers in degrees. Should be in the
%             interval [70,180], otherwise default scene is chosen
%
%  Output arguments: 
%  CalM     - 9x3 matrix containing the M calibration 3x3 matrices for 
%             each camera concatenated.
%  R_t      - 3-cell containing two 3x4 orientation matrices [R2,t2]
%             and [R3,t3], the first camera is [Id,0].
%  Corresp  - 6xN matrix containing in each column, the 3 projections of
%             the same space point onto the 3 images.
%  points3D - 3xN matrix containing the 3D points.
%
%%% measurements are in mm
%%% sensor size 36x24mm --> image size 1800x1200 pixels 
%%% (resolution 1mm=50pix)

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

if isempty(angle) || angle<70 || angle>180
    p_coll=0; % default setting
else
    angle=angle*pi./180;  % to radians
    p_coll= 1- sin(angle)/(sqrt(2)*(cos(angle)-1));
end

%%% Calibration matrix (depends on the focal length)
k=focalL/50; % focal length factor
pix=50;      % number of pixels in 1mm (resolution)
K=[50*k*pix     0   18*pix;...
        0 50*k*pix  12*pix;...
        0       0       1];

%%% Camera centers, depending on collinear factor and focal length
C1=k*[   0;-1400; 400] + k*p_coll*[0; 300;-300];
C2=k*[-400;-1000;   0] + k*p_coll*[0;-100; 100];
C3=k*[ 600; -800;-200] + k*p_coll*[0;-300; 300];

%%% Rotation matrices, making the cameras point to the center of coord. (0,0,0)
R1=rotation(C1,[0;0;-1]);
R2=rotation(C2,[0;0;-1]);
R3=rotation(C3,[0;0;-1]);

%%% Projection matrices normalized
P1=K*R1*[eye(3) -C1];   P1=P1*sqrt(24)/norm(P1);
P2=K*R2*[eye(3) -C2];   P2=P2*sqrt(24)/norm(P2);
P3=K*R3*[eye(3) -C3];   P3=P3*sqrt(24)/norm(P3);

%%% generation&projection of N random 3D points
rng(seed)
M=N;
points3D=zeros(3,N);
Corresp=zeros(6,N);
ind1=0;
while M>0;
    % 3D points generation
    X=400*rand(3,M)-200;

    % project 3D points
    x1=P1*[X; ones(1,M)];   x1=x1./repmat(x1(3,:),3,1);
    x2=P2*[X; ones(1,M)];   x2=x2./repmat(x2(3,:),3,1);
    x3=P3*[X; ones(1,M)];   x3=x3./repmat(x3(3,:),3,1);
    
    % add noise
    x1_noise = x1(1:2,:)+randn(2,M)*noise;
    x2_noise = x2(1:2,:)+randn(2,M)*noise;
    x3_noise = x3(1:2,:)+randn(2,M)*noise;

    % find image points inside image limits
    inside=find(x1_noise(1,:)<=36*pix & x1_noise(2,:)<= 24*pix &...
        x2_noise(1,:)<=36*pix & x2_noise(2,:)<= 24*pix &...
        x3_noise(1,:)<=36*pix & x3_noise(2,:)<= 24*pix &...
        x1_noise(1,:)>=0 & x1_noise(2,:)>= 0 &...
        x2_noise(1,:)>=0 & x2_noise(2,:)>= 0 &...
        x3_noise(1,:)>=0 & x3_noise(2,:)>= 0);
    
    Corresp(:,ind1+(1:length(inside)))=[x1_noise(:,inside);...
        x2_noise(:,inside); x3_noise(:,inside)];
    
    points3D(:,ind1+(1:length(inside)))=X(:,inside);
    
    ind1=ind1+length(inside);
    
    % remaining points to generate
    M=N-ind1;
end

R_t={R2*[R1.' (C1-C2)], R3*[R1.' (C1-C3)]};

CalM=repmat(K,3,1);

end

function R=rotation(u,v)
if size(u,1)==1
    u=u.';
end
if size(v,1)==1
    v=v.';
end
u=u/norm(u);    v=v/norm(v);
w=cross(u,v);   
s=norm(w);
c=dot(u,v);
%C=[0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
%R=eye(3)+C+((1-c)/(s^2))*C^2;
w=w/s;
R=c*eye(3)+s*crossM(w)+(1-c)*(w*w.');

end





