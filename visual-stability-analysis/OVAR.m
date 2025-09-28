% Read CSV file information
% The CSV contains 6 columns: {'framenumber','framerate','ledparameter','pupx','pupy','lightspotx','lightspoty'}
csvdata = readtable(path);
Fs = csvdata.framerate(1);  % Extract frame rate (use the first value as frame rate is constant)

% Extract data: Subtract light spot coordinates from pupil coordinates, then apply median filtering
pupy = medfilt1(csvdata.pupy - csvdata.lightspoty, ceil(10/hfre0));  % Y-axis pupil data filtering
pupx = medfilt1(csvdata.pupx - csvdata.lightspotx, ceil(10/hfre0));  % X-axis pupil data filtering
% pupx = smooth(pupx,20);  % Alternative smoothing method (commented out)

% Remove the first 100 frames (initialization/transient data)
pupy = pupy(100:end);
pupx = pupx(100:end);

% Calculate pupil distance (Euclidean distance of X/Y pupil coordinates)
for i = 1 : length(pupx)
    pupd(i) = sqrt(pupx(i)^2 + pupy(i)^2);
end

% Calculate time vector: Convert frame numbers to actual time (frame number / frame rate)
xtime = csvdata.framenumber(100:end) / csvdata.framerate(1);
% % figure,plot(xtime,pupx)  % Optional plot for X-axis pupil data (commented out)

% Detect peaks in LED parameter data to determine stimulation frequency
% 'MinPeakDistance' sets minimum interval between consecutive peaks
[peaky1, peakx1] = findpeaks(csvdata.ledparameter, 'MinPeakDistance', csvdata.framerate(1)/(1.4*hfre0));  % LED on/off peak info
hfre = Fs / mean(diff(peakx1));  % Calculate actual stimulation frequency (hfre)

% Create time vector normalized by minutes (for velocity plot)
TT = length(pupx)/60;  % Total duration in minutes
tt = linspace(0, TT, length(pupx));  % Generate evenly spaced time points
tt = tt';  % Convert to column vector


% Create figure with 2x4 subplots for data visualization
figure;
% Subplot 242: Y-axis pupil data + detected peaks
subplot(242)
plot(pupy)
[peaky, peakx] = findpeaks(pupy, 'MinPeakDistance', csvdata.framerate(1)/(1.4*hfre));
hold on
plot(peakx, peaky, '*')  % Mark peaks with asterisks
title(sprintf("%sYPeaks", mousename))  % Title with mouse ID

% Subplot 241: X-axis pupil raw data
subplot(241)
plot(pupx)
title('XAxis')

% Subplot 245: X-axis pupil velocity (1st derivative of pupil position)
subplot(245)
v = diff(csvdata.pupx(99:length(csvdata.pupx)));  % Compute derivative (adjust index for length match)
plot(tt, v)
title(sprintf("%sXAxisV", mousename))  % Title with mouse ID
% xlim([0 30])  % Optional x-axis limit (commented out)

% Subplot 246: Pupil distance (Euclidean) + detected peaks
subplot(246)
plot(pupd)
[peaky_d, peakx_d] = findpeaks(pupd, 'MinPeakDistance', csvdata.framerate(1)/(1.4*hfre));
hold on
plot(peakx_d, peaky_d, '*')  % Mark peaks with asterisks
title(sprintf("%sXYPeaks", mousename))  % Title with mouse ID


%% Segment selection based on detected peaks (for Y-axis pupil data)
% Initialize matrix to store start/end indices of valid peak segments
finalpointsind = [];
finalpointsind(1,1) = 1;  % 1st row: start indices of segments
finalpointsind(2,1) = 2;  % 2nd row: end indices of segments
count = 1;  % Counter for segment number

% Threshold-based segment grouping (commented out alternative logic)
% if mean(pupx)>900
%     yyind = 0.01;
% elseif mean(pupx)<900
%      yyind = 0.009;
% end

% Alternative segment grouping logic (commented out)
% for ind = 3:length(peakx)
%     tempy2 = peaky(finalpointsind(2,count));
%     tempy1 = peaky(finalpointsind(1,count));
%     if abs(peaky(ind)-tempy1) <  hamp  && abs(peaky(ind)-tempy2) <  hamp 
% %         (yyind*tempy)
%             finalpointsind(2,count) = ind;
%     else 
%           count = count+1;
%           finalpointsind(1,count) = ind;
%           finalpointsind(2,count) = ind+1;
%     end
% end

% Current segment grouping logic: Group peaks by amplitude similarity to segment start
for ind = 2:length(peakx)
    tempy1 = peaky(finalpointsind(1,count));  % Amplitude of current segment's start peak
    % If current peak amplitude is within 'hamp' threshold of start peak
    if abs(peaky(ind) - tempy1) < hamp  
%         (yyind*tempy)  % Placeholder for alternative threshold (commented out)
        finalpointsind(2,count) = ind;  % Extend current segment to include this peak
    else 
        count = count + 1;  % Start new segment
        finalpointsind(1,count) = ind;  % New segment start index
        finalpointsind(2,count) = ind + 1;  % New segment end index (initial)
    end
end

% Filter out invalid segments (length < 2 peaks)
% odev = temp(2:2:size(temp,1),3)-temp(1:2:size(temp,1),3);  % Commented out unused logic
% odevidn = find(odev>=3);  % Commented out unused logic
% out =[temp(2*odevidn-1,:);temp(2*odevidn,:)];  % Commented out unused logic
stind = finalpointsind(1,:);  % Extract all segment start indices
seind = finalpointsind(2,:);  % Extract all segment end indices
tx = seind - stind;  % Calculate length of each segment (number of peaks)
txind = find(tx < 2);  % Find segments with <2 peaks (invalid)
stind(txind) = [];  % Remove invalid start indices
seind(txind) = [];  % Remove invalid end indices
% stind = stind+1;  % Optional index adjustment (commented out)
% seind = seind-1;  % Optional index adjustment (commented out)

% Initialize arrays to store FFT results (amplitude and frequency)
amp = [];  
amp2 = [];
%% FFT analysis for valid Y-axis pupil segments
for inst = 1:length(stind)
    % Get start/end frame indices of current segment (from peak indices)
    t1 = peakx(stind(inst));
    t2 = peakx(seind(inst));
    
    % Adjust end index to ensure integer number of stimulation periods
    numofperiod = floor((t2 - t1) .* hfre ./ Fs);  % Calculate number of full periods
    t2 = t1 + (numofperiod / hfre) * Fs;  % Update end index to match full periods
    numofperiod = floor((t2 - t1) .* hfre ./ Fs);  % Recheck (redundant but retains original logic)
    t2 = t1 + (numofperiod / hfre) * Fs;  
    
    % Extract Y-axis pupil data for current segment
    nb = pupy(t1:t2);
    
    % FFT parameter setup
    T = 1/Fs;  % Time resolution (sampling interval)
    L = length(nb);  % Number of data points in segment
    t = (0:L-1)*T;  % Time vector for current segment
    nb = nb - mean(nb);  % Remove DC component (zero-mean normalization)
    N = 2^nextpow2(L);  % Use next power of 2 for faster FFT computation
    Y = fft(nb, N)/N * 2;  % FFT with amplitude normalization (excluding DC)
    f = Fs/N*(0:1:N-1);  % Frequency vector
    A = abs(Y);  % Extract amplitude from complex FFT result
    
    % Keep only positive frequency components (Nyquist frequency)
    x = f(1:N/2);
    y = A(1:N/2);
    yt = 1:length(y);  % Placeholder for index (retains original logic)
    k = 1;  % Placeholder for sorting (retains original logic)
    
    % Find peak amplitude within a specific frequency range [maxx(1), maxx(2)]
    xx1 = sort([x, maxx(1), maxx(2)]);  % Sort frequencies + range bounds
    I(1) = find(xx1 == maxx(1));  % Start index of target frequency range
    I(2) = find(xx1 == maxx(2)) - 1;  % End index of target frequency range
    [maxy, ~] = max(y(I(1):I(2)));  % Maximum amplitude in target range
    % Corresponding frequency of maximum amplitude (adjust index for range offset)
    maxxx = x(find(y(I(1):I(2)) == max(y(I(1):I(2)))) + I(1) - 1);
    
    % Store FFT results
    data1 = maxxx; 
    data2 = maxy;
    amp(inst) = maxy;  % Store maximum amplitude of current segment
    amp2(inst) = maxxx;  % Store corresponding frequency of current segment
    
    % Plot segment boundaries on Y-axis pupil data
    subplot(243)
    plot(pupy)
    hold on
    plot(peakx(stind), peaky(stind), '*')  % Mark segment start peaks
    hold on
    plot(peakx(seind), peaky(seind), 'o')  % Mark segment end peaks
    title(sprintf("%sYAxis", mousename))  % Title with mouse ID
    
    % Plot FFT amplitude spectrum (Y-axis pupil data)
    subplot(244)
    plot(x, y)
    xlim([0 1])  % Limit X-axis to 0-1 Hz (focus on stimulation frequency)
    ylim([0 50])  % Limit Y-axis for consistent visualization
    title(sprintf("%sFre", mousename))  % Title with mouse ID
end


%% Segment selection based on detected peaks (for pupil distance data)
% Re-initialize segment index matrix (for pupil distance peaks)
finalpointsind = [];
finalpointsind(1,1) = 1;  % 1st row: start indices
finalpointsind(2,1) = 2;  % 2nd row: end indices
count = 1;  % Reset segment counter

% Threshold-based segment grouping (commented out alternative logic)
% if mean(pupx)>900
%     yyind = 0.01;
% elseif mean(pupx)<900
%      yyind = 0.009;
% end

% Alternative segment grouping logic (commented out)
% for ind = 3:length(peakx)
%     tempy2 = peaky(finalpointsind(2,count));
%     tempy1 = peaky(finalpointsind(1,count));
%     if abs(peaky(ind)-tempy1) <  hamp  && abs(peaky(ind)-tempy2) <  hamp 
% %         (yyind*tempy)
%             finalpointsind(2,count) = ind;
%     else 
%           count = count+1;
%           finalpointsind(1,count) = ind;
%           finalpointsind(2,count) = ind+1;
%     end
% end

% Current segment grouping logic: Group pupil distance peaks by amplitude similarity
for ind = 2:length(peakx_d)
    tempy1 = peaky_d(finalpointsind(1,count));  % Amplitude of current segment's start peak
    % If current peak amplitude is within 'hampd' threshold of start peak
    if abs(peaky_d(ind) - tempy1) < hampd 
%         (yyind*tempy)  % Placeholder for alternative threshold (commented out)
        finalpointsind(2,count) = ind;  % Extend current segment
    else 
        count = count + 1;  % Start new segment
        finalpointsind(1,count) = ind;  % New segment start index
        finalpointsind(2,count) = ind + 1;  % New segment end index (initial)
    end
end

% Filter out invalid segments (length < 2 peaks)
% odev = temp(2:2:size(temp,1),3)-temp(1:2:size(temp,1),3);  % Commented out unused logic
% odevidn = find(odev>=3);  % Commented out unused logic
% out =[temp(2*odevidn-1,:);temp(2*odevidn,:)];  % Commented out unused logic
stind = finalpointsind(1,:);  % Extract valid start indices
seind = finalpointsind(2,:);  % Extract valid end indices
tx = seind - stind;  % Calculate segment length (number of peaks)
txind = find(tx < 2);  % Find invalid segments (<2 peaks)
stind(txind) = [];  % Remove invalid starts
seind(txind) = [];  % Remove invalid ends
% stind = stind+1;  % Optional index adjustment (commented out)
% seind = seind-1;  % Optional index adjustment (commented out)

% Initialize arrays to store FFT results (pupil distance)
amp_d = [];  
amp2_d = [];
%% FFT analysis for valid pupil distance segments
for inst = 1:length(stind)
    % Get start/end frame indices of current segment (from peak indices)
    t1 = peakx_d(stind(inst));
    t2 = peakx_d(seind(inst));
    
    % Adjust end index to ensure integer number of stimulation periods
    numofperiod_d = floor((t2 - t1) .* hfre ./ Fs);  % Number of full periods
    t2 = t1 + (numofperiod_d / hfre) * Fs;  % Update end index
    numofperiod_d = floor((t2 - t1) .* hfre ./ Fs);  % Recheck (retains original logic)
    t2 = t1 + (numofperiod_d / hfre) * Fs;  
    
    % Extract pupil distance data for current segment
    nb = pupd(t1:t2);
    
    % FFT parameter setup (same as Y-axis analysis)
    T = 1/Fs;
    L = length(nb);
    t = (0:L-1)*T;
    nb = nb - mean(nb);  % Zero-mean normalization
    N = 2^nextpow2(L);  % FFT optimization
    Y = fft(nb, N)/N * 2;  % FFT with amplitude normalization
    f = Fs/N*(0:1:N-1);
    A = abs(Y);
    
    % Keep positive frequency components
    x = f(1:N/2);
    y = A(1:N/2);
    yt = 1:length(y);  % Placeholder (retains original logic)
    k = 1;  % Placeholder (retains original logic)
    
    % Find peak amplitude in target frequency range [maxx(1), maxx(2)]
    xx1 = sort([x, maxx(1), maxx(2)]);
    I(1) = find(xx1 == maxx(1));  % Start of target range
    I(2) = find(xx1 == maxx(2)) - 1;  % End of target range
    [maxy, ~] = max(y(I(1):I(2)));  % Max amplitude in range
    % Corresponding frequency (adjust index for range offset)
    maxxx = x(find(y(I(1):I(2)) == max(y(I(1):I(2)))) + I(1) - 1);
    
    % Store FFT results for pupil distance
    data1 = maxxx; 
    data2 = maxy;
    amp_d(inst) = maxy;  % Max amplitude of current segment
    amp2_d(inst) = maxxx;  % Corresponding frequency
    
    % Plot segment boundaries on pupil distance data
    subplot(247)  
    plot(pupd)
