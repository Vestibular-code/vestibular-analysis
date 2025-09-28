% Read CSV information
% The table contains six columns: {'framenumber','framerate','ledparameter','pupx','pupy','lightspotx','lightspoty'}
csvdata = readtable(path);
Fs  = csvdata.framerate(1);

% Extract some data: Subtract the light spot data from the pupil data and apply a simple median filter
pupx = medfilt1(csvdata.pupx - csvdata.lightspotx, ceil(10/hfre0));  
% pupx = smooth(pupx,20); % Alternative smoothing method (commented out)

% Remove the first 100 frames of data
pupx = pupx(100:end);

% Calculate time by dividing frame number by frame rate
xtime = csvdata.framenumber(100:end) / csvdata.framerate(1);
% figure,plot(xtime,pupx) % Optional plot (commented out)

% Smooth the LED parameter data
leddata = smooth(csvdata.ledparameter, 10);
leddata = leddata(100:end);

% Find peaks in the LED data to determine the stimulation frequency
[peaky1, peakx1] = findpeaks(leddata, 'MinPeakDistance', csvdata.framerate(1) / (1.4 * hfre0)); % LED light on/off peak information
hfre = Fs / mean(diff(peakx1));
% figure; plot(csvdata.ledparameter) % Optional plot (commented out)
% figure; plot(smooth(csvdata.ledparameter,15)) % Optional plot (commented out)


% Plot the pupil data and its peaks
figure,
subplot(221)
plot(pupx)
[peaky, peakx] = findpeaks(pupx, 'MinPeakDistance', csvdata.framerate(1) / (1.4 * hfre));
hold on
plot(peakx, peaky, '*')

%%  After obtaining peak data, select segments from it
% Initialize variables to store segment indices
finalpointsind = [];
finalpointsind(1,1) = 1;
finalpointsind(2,1) = 2;
count = 1;

% Determine a threshold based on mean pupil size (commented out)
% if mean(pupx)>900
%     yyind = 0.01;
% elseif mean(pupx)<900
%      yyind = 0.009;
% end

% Loop through peaks to group them into segments
for ind = 3:length(peakx)
    tempy2 = peaky(finalpointsind(2,count));
    tempy1 = peaky(finalpointsind(1,count));
    
    % If current peak is close in amplitude to both ends of the current segment
    if abs(peaky(ind) - tempy1) < hamp && abs(peaky(ind) - tempy2) < hamp 
        % Extend the current segment to include this peak
        finalpointsind(2,count) = ind;
    else 
        % Start a new segment
        count = count + 1;
        finalpointsind(1,count) = ind;
        finalpointsind(2,count) = ind + 1;
    end
end

% Extract start and end indices of valid segments
stind = finalpointsind(1,:);
seind = finalpointsind(2,:);

% Remove segments that are too short (less than 2 peaks)
tx = seind - stind;
txind = find(tx < 2);
stind(txind) = [];
seind(txind) = [];
% stind = stind + 1; % Optional adjustment (commented out)
% seind = seind - 1; % Optional adjustment (commented out)

% Initialize variables to store amplitude and frequency results
amp = []; 
amp2 = [];
%% 

% Process each selected segment using FFT
for inst = 1:length(stind)
    % Get the start and end peak indices for the current segment
    t1 = peakx(stind(inst));
    t2 = peakx(seind(inst));
    
    % Adjust the end index to ensure it covers an integer number of stimulation periods
    numofperiod = floor((t2 - t1) .* hfre ./ Fs);
    t2 = t1 + (numofperiod / hfre) * Fs;
    
    % Extract the pupil data for this segment
    nb = pupx(t1:t2);

    % Prepare data for FFT
    T = 1 / Fs;
    L = length(nb);
    t = (0:L-1) * T;
    nb = nb - mean(nb);    % Remove DC component (zero-mean)
    
    % Perform FFT
    N = 2^nextpow2(L); % Use next power of 2 for faster FFT
    Y = fft(nb, N) / N * 2; % Fast Fourier Transform and normalize
    f = Fs / N * (0:1:N-1); 
    A = abs(Y);    
    x = f(1:N/2);
    y = A(1:N/2);
    
    % Find the maximum amplitude and corresponding frequency within a specific range
    yt = 1:length(y);
    k = 1;               % Placeholder for sorting (not directly used)
    xx1 = sort([x, maxx(1), maxx(2)]);
    I(1) = find(xx1 == maxx(1));
    I(2) = find(xx1 == maxx(2)) - 1;
    
    [maxy, ~] = max(y(I(1):I(2))); % Find the maximum y value in the interval
    maxxx = x(find(y(I(1):I(2)) == max(y(I(1):I(2)))) + I(1) - 1); % Find the corresponding x value
    
    % Store results
    data1 = maxxx; 
    data2 = maxy;
    amp(inst) = maxy;
    amp2(inst) = maxxx;
    
    % Plotting
    subplot(222)
    plot(pupx)
    hold on
    plot(peakx(stind), peaky(stind), '*') % Mark segment start points
    hold on
    plot(peakx(seind), peaky(seind), 'o') % Mark segment end points
    
    subplot(223),
    plot(x, y) % Plot the frequency spectrum
end

%% Calculate the gain value
w = 20; % Input the mouse's weight, default is 20g
radius = 1/2 * (0.02026 * w + 2.611); % Relationship between mouse weight and eyeball size (mm)

output = mean(amp);       % Amplitude in pixels
bi = output / (239.24 * 1); % Convert pixels to millimeters
bi = asin(bi);
angumouse = bi * 180 / pi; % Mouse eye movement angle in degrees

anguep = str2double(velocity) / (2 * pi * hfre);    % Rotation angle of the platform in degrees

gain = angumouse / anguep; % Calculate the gain
