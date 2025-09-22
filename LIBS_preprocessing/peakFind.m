function [nmMaxLoc, pixMaxLoc, nmPeakRange, pixPeakRange] = ...
    peakFind(spectra, num, minH, minDiff, pixel, smoo, plotMinMax)
% Ѱ���׷� Find certain number of peaks in spectra
% ���� Input:
%     spectra:
%         pixNum �� (1 + specNum)
%         (:, 1) - �����صĹ��� wavelength of each pixel
%         (:, 2 : end) - ���ж�Ӧһ���� each column is a spectrum
%     num:
%         ����׷����� the maximum number of peaks
%     minH:
%         ��ʶ��Ϊ�׷����С�߶� 
%         defines the minimum height to be recognized as a peak
%         for the whole spectra
%         height > threshold = (max - min) * minH + min
%     minDiff:
%         ��ʶ��Ϊ�׷����С�߲�
%         defines the minimum difference to be recognized as a peak
%         for a peak
%         min < threshold = max * minDiff
%     pixel:
%         ������
%         the maximum width of half a peak in pixels
%     smoo:
%         ƽ����Χ
%         the span of spectra smoothing
%     plotMinMax:
%         �Ƿ񻭳�����ƽ��������ص���С�����ֵ
%         if the minimum and maximum of each pixel should be plotted
% 
% ��� Output:
%     nmMaxLoc:
%         ���׷�λ�ã������� the location of peak values in nm
%     pixMaxLoc:
%         ���׷�λ�ã����أ� the location of peak values in pixel number
%     nmPeakRange:
%         peakNum �� 2
%         ���׷巶Χ�������� the range of each peak in nm
%     pixPeakRange:
%         peakNum �� 2
%         ���׷巶Χ�����أ� the range of each peak in pixel number
%  ���ߣ������࣬�����
%  �޸�ʱ�䣺2022.03.16��11:35
%  ��ϵ��ʽ��gwl1997@foxmail.com

% ����ƽ��
if smoo > 0
    for i = 2 : size(spectra, 2)
        spectra(:, i) = smooth(spectra(:, i), smoo);
    end
end
data = mean(spectra(:, 2 : end), 2);   % ƽ������ averaged spectra
thres = (max(data) - min(data)) * minH + min(data); % ��С��� minimum height of a peak
wavelength = spectra(:, 1);
% ���ҹ��׼���ֵ����Сֵλ��
[maxV, maxLoc]= findpeaks(data, 'MinPeakHeight', thres); %'SORTSTR','descend'
[minV, minLoc]= findpeaks(-1 * data);
peakNum = min(num, length(maxLoc));
pixPeakRange = []; pixMaxLoc = [];
[~, ind] = sort(maxV, 'descend'); % ������ֵ��С�������
j = 1;
for i = 1 : length(maxLoc)
    % �ɼ�Сֵȷ���׷巶Χ determine the peak range based on minimums
    temp = find(minLoc < maxLoc(ind(i))); 
    if isempty(temp)
        tempRange = [1, minLoc(1)];
    elseif length(minLoc) == temp(end)
        tempRange = [minLoc(end), length(wavelength)];
    else
        tempRange = [minLoc(temp(end)), minLoc(temp(end) + 1)];
    end
    % ����������ü��׷巶Χ narrow the range based on "pixel"
    tempRange2 = tempRange;
    if pixel ~= 0
        if maxLoc(ind(i)) - pixel > tempRange(1)
            tempRange2(1) = maxLoc(ind(i)) - pixel;
        end
        if maxLoc(ind(i)) + pixel < tempRange(2)
            tempRange2(2) = maxLoc(ind(i)) + pixel;
        end
    end
    % �ж��Ƿ�������С�߲�Ҫ��
    if min(data(tempRange)) < minDiff * maxV(ind(i))
        pixPeakRange = [pixPeakRange; tempRange2];
        pixMaxLoc = [pixMaxLoc; maxLoc(ind(i))];
        j = j + 1;
        if j > peakNum
            break;
        end
    end
end

% ��ֵ�����׷巶Χ����ֵλ�ã������� calculate the peak range and max loc (nm)
nmPeakRange = interp1((1:size(wavelength, 1))',...
    wavelength, pixPeakRange, 'nearest', 'extrap');
nmMaxLoc = interp1((1:size(wavelength, 1))',...
    wavelength, pixMaxLoc, 'nearest', 'extrap');

% ��ͼ������ - ƽ�����ף���+ - �׷�λ�ã���+ - �׷巶Χ
figure; hold on;
plot(wavelength, data, 'k-');
plot(nmMaxLoc, data(pixMaxLoc), 'r+');
plot(nmPeakRange(:, 1), data(pixPeakRange(:, 1)), 'b+');
plot(nmPeakRange(:, 2), data(pixPeakRange(:, 2)), 'b+');

if plotMinMax
    plot(wavelength, min(spectra(:, 2 : end), [], 2), 'color', [0.7 0.7 0.7]);
    plot(wavelength, max(spectra(:, 2 : end), [], 2), 'color', [0.7 0.7 0.7]);
end
end

