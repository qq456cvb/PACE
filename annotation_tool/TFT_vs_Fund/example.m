% Example on how to use the code of the repository TFT_vs_Fund

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

clear;

%% Generate some random data for a triplet of images
N=100;      % number of 3D points
noise=1;    % sigma for the added Gaussian noise in pixels
seed=1;     % seed for random generation
f=50;       % focal length in mm
angle=0;    % default angle (no collinearity of camera centers)

[CalM,R_t0,Corresp,points3D]=generateSyntheticScene(N,noise,seed,f,angle);

%% Test a method

method={...
    @LinearTFTPoseEstimation,...    % 1 - TFT - Linear estimation
    @ResslTFTPoseEstimation,...     % 2 - TFT - Ressl
    @NordbergTFTPoseEstimation,...  % 3 - TFT - Nordberg
    @FaugPapaTFTPoseEstimation,...  % 4 - TFT - Faugeras&Papadopoulo 
    @PiPoseEstimation,...           % 5 - Pi matrices - Ponce&Hebert
    @PiColPoseEstimation,...        % 6 - Pi matrices - Ponce&Hebert for collinear cameras
    @LinearFPoseEstimation,...      % 7 - Fundamental matrices - Linear estimation
    @OptimFPoseEstimation};         % 8 - Fundamental matrices - Optimized

[R_t_2,R_t_3,Reconst]=method{4}(Corresp,CalM);

%% Compute the errors

% reprojection error
repr_err=ReprError({CalM(1:3,:)*eye(3,4),CalM(4:6,:)*R_t_2,CalM(7:9,:)*R_t_3},...
    Corresp,Reconst);
fprintf('Reprojection error is %f .\n',repr_err);

% angular errors
[rot2_err,t2_err]=AngError(R_t0{1},R_t_2);
[rot3_err,t3_err]=AngError(R_t0{2},R_t_3);

fprintf('Angular errors in rotations are %f and %f, and in translations are %f and %f .\n',...
    rot2_err,rot3_err,t2_err,t3_err);

%% Apply Bundle Adjustment

[R_t_ref,Reconst_ref,iter,repr_err]=BundleAdjustment(CalM,[eye(3,4);R_t_2;R_t_3],Corresp,Reconst);
fprintf('Reprojection error is %f after Bundle Adjustment.\n',repr_err);

% angular errors
[rot2_err,t2_err]=AngError(R_t0{1},R_t_ref(4:6,:));
[rot3_err,t3_err]=AngError(R_t0{2},R_t_ref(7:9,:));
fprintf('Angular errors in rotations after BA are %f and %f, and in translations are %f and %f .\n',...
    rot2_err,rot3_err,t2_err,t3_err);




