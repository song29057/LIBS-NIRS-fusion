function [peakIntensity] = ...
    peakCalculate(wavelength, spectra, pixPeakRange)
% Calculate the intensity of the peaks
% 
% Input:
%     wavelenth:
%         nPixel ¡Á 1
%     spectra:
%         nPixel ¡Á specNum
%     pixPeakRange:
%         nPeak ¡Á 2
%         the range of each peak in pixel
% 
% Output:
%     peakIntensity:
%         nPeak ¡Á specNum
%         the intensity of each peak

peakNum = size(pixPeakRange, 1);
peakIntensity = zeros(peakNum, size(spectra, 2));
minimum = min(0, min(min(spectra)));
spectra = spectra - minimum;
for i = 1 : peakNum
    range = pixPeakRange(i, 1) : pixPeakRange(i, 2);    
    if size(spectra, 2) > 1
%         peakIntensity(i, :) = trapz(wavelength(range), spectra(range, :)) ...
%             - 1 / 2 * (wavelength(range(end)) - wavelength(range(1))) ...
%             * (spectra(range(1), :) + spectra(range(end), :));
        peakIntensity(i, :) = sum(spectra(range, :)) - ...
            1 / 2 * length(range) * (spectra(range(1), :) + spectra(range(end), :));
%         peakIntensity(i, :) = sum(spectra(range, :));
    else
%         peakIntensity(i) = trapz(wavelength(range), spectra(range)) ...
%             - 1 / 2 * (wavelength(range(end)) - wavelength(range(1))) ...
%             * (spectra(range(1)) + spectra(range(end)));
        peakIntensity(i, :) = sum(spectra(range)) - ...
            1 / 2 * length(range) * (spectra(range(1)) + spectra(range(end)));
%         peakIntensity(i, :) = sum(spectra(range));
    end
end
end