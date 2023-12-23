function [K,R,t,im_size]=readCalibrationOrientation_EPFL(image_path,image_name)
%READCALIBRATIONORIENTATION_EPFL Gets the true calibration and orientation
% of a camera in an EPFL dataset.

filename=strcat(image_path,image_name,'.camera');
calibration_file = fopen(filename,'r');

K=[str2num(fgetl(calibration_file));...
    str2num(fgetl(calibration_file));...
    str2num(fgetl(calibration_file))];

fgetl(calibration_file);

R=[str2num(fgetl(calibration_file));...
    str2num(fgetl(calibration_file));...
    str2num(fgetl(calibration_file))].';

t=-R*[str2num(fgetl(calibration_file))].';

im_size=[str2num(fgetl(calibration_file))].';

fclose(calibration_file);

end


