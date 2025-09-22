function [nmMaxLoc, pixMaxLoc, nmPeakRange, pixPeakRange] = ...
    peakFind(spectra, num, minH, minDiff, pixel, smoo, plotMinMax)
% 寻找谱峰 Find certain number of peaks in spectra
% 输入 Input:
%     spectra:
%         pixNum × (1 + specNum)
%         (:, 1) - 各像素的光谱 wavelength of each pixel
%         (:, 2 : end) - 各列对应一光谱 each column is a spectrum
%     num:
%         最大谱峰数量 the maximum number of peaks
%     minH:
%         可识别为谱峰的最小高度 
%         defines the minimum height to be recognized as a peak
%         for the whole spectra
%         height > threshold = (max - min) * minH + min
%     minDiff:
%         可识别为谱峰的最小高差
%         defines the minimum difference to be recognized as a peak
%         for a peak
%         min < threshold = max * minDiff
%     pixel:
%         最大半峰宽
%         the maximum width of half a peak in pixels
%     smoo:
%         平滑范围
%         the span of spectra smoothing
%     plotMinMax:
%         是否画出光谱平滑后各像素点最小、最大值
%         if the minimum and maximum of each pixel should be plotted
% 
% 输出 Output:
%     nmMaxLoc:
%         各谱峰位置（波长） the location of peak values in nm
%     pixMaxLoc:
%         各谱峰位置（像素） the location of peak values in pixel number
%     nmPeakRange:
%         peakNum × 2
%         各谱峰范围（波长） the range of each peak in nm
%     pixPeakRange:
%         peakNum × 2
%         各谱峰范围（像素） the range of each peak in pixel number
%  作者：侯宗余，顾炜伦
%  修改时间：2022.03.16，11:35
%  联系方式：gwl1997@foxmail.com

% 光谱平滑
if smoo > 0
    for i = 2 : size(spectra, 2)
        spectra(:, i) = smooth(spectra(:, i), smoo);
    end
end
data = mean(spectra(:, 2 : end), 2);   % 平均光谱 averaged spectra
thres = (max(data) - min(data)) * minH + min(data); % 最小峰高 minimum height of a peak
wavelength = spectra(:, 1);
% 查找光谱极大值、极小值位置
[maxV, maxLoc]= findpeaks(data, 'MinPeakHeight', thres); %'SORTSTR','descend'
[minV, minLoc]= findpeaks(-1 * data);
peakNum = min(num, length(maxLoc));
pixPeakRange = []; pixMaxLoc = [];
[~, ind] = sort(maxV, 'descend'); % 按极大值大小降序查找
j = 1;
for i = 1 : length(maxLoc)
    % 由极小值确定谱峰范围 determine the peak range based on minimums
    temp = find(minLoc < maxLoc(ind(i))); 
    if isempty(temp)
        tempRange = [1, minLoc(1)];
    elseif length(minLoc) == temp(end)
        tempRange = [minLoc(end), length(wavelength)];
    else
        tempRange = [minLoc(temp(end)), minLoc(temp(end) + 1)];
    end
    % 根据最大半峰宽裁剪谱峰范围 narrow the range based on "pixel"
    tempRange2 = tempRange;
    if pixel ~= 0
        if maxLoc(ind(i)) - pixel > tempRange(1)
            tempRange2(1) = maxLoc(ind(i)) - pixel;
        end
        if maxLoc(ind(i)) + pixel < tempRange(2)
            tempRange2(2) = maxLoc(ind(i)) + pixel;
        end
    end
    % 判断是否满足最小高差要求
    if min(data(tempRange)) < minDiff * maxV(ind(i))
        pixPeakRange = [pixPeakRange; tempRange2];
        pixMaxLoc = [pixMaxLoc; maxLoc(ind(i))];
        j = j + 1;
        if j > peakNum
            break;
        end
    end
end

% 插值计算谱峰范围及峰值位置（波长） calculate the peak range and max loc (nm)
nmPeakRange = interp1((1:size(wavelength, 1))',...
    wavelength, pixPeakRange, 'nearest', 'extrap');
nmMaxLoc = interp1((1:size(wavelength, 1))',...
    wavelength, pixMaxLoc, 'nearest', 'extrap');

% 画图，黑线 - 平均光谱，红+ - 谱峰位置，蓝+ - 谱峰范围
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

