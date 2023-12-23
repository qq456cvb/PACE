function T_new = transform_TFT(T_old,M1,M2,M3,inverse)
% Tranformed TFT
%
% short function to transform the TFT when an algebraic transformation
% has been aplied to the image points.
%
% if inverse==0 :
%   from a TFT T_old assossiated to P1_old, P2_old, P3_old, find the new TFT
%   T_new associated to P1_new=M1*P1_old, P2_new=M2*P2_old, P3_new=M3*P3_old.
% if inverse==1 :
%   from a TFT T_old assossiated to P1_old, P2_old, P3 _old, find the new TFT
%   T_new associated to M1*P1_new=P1_old, M2*P2_new=P2_old, M3*P3_new=P3_old.
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


if nargin<5
    inverse=0;
end

if inverse==0
    M1i=inv(M1); T_new=zeros(3,3,3);
    T_new(:,:,1)=M2*(M1i(1,1)*T_old(:,:,1) + M1i(2,1)*T_old(:,:,2) + M1i(3,1)*T_old(:,:,3) )*M3.';
    T_new(:,:,2)=M2*(M1i(1,2)*T_old(:,:,1) + M1i(2,2)*T_old(:,:,2) + M1i(3,2)*T_old(:,:,3) )*M3.';
    T_new(:,:,3)=M2*(M1i(1,3)*T_old(:,:,1) + M1i(2,3)*T_old(:,:,2) + M1i(3,3)*T_old(:,:,3) )*M3.';

elseif inverse==1
    M2i=inv(M2); M3i=inv(M3); T_new=zeros(3,3,3);
    T_new(:,:,1)=M2i*(M1(1,1)*T_old(:,:,1) + M1(2,1)*T_old(:,:,2) + M1(3,1)*T_old(:,:,3) )*M3i.';
    T_new(:,:,2)=M2i*(M1(1,2)*T_old(:,:,1) + M1(2,2)*T_old(:,:,2) + M1(3,2)*T_old(:,:,3) )*M3i.';
    T_new(:,:,3)=M2i*(M1(1,3)*T_old(:,:,1) + M1(2,3)*T_old(:,:,2) + M1(3,3)*T_old(:,:,3) )*M3i.';
end

T_new=T_new/norm(T_new(:));

end
