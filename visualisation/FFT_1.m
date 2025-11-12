subjects = 8; % Number of subjects
tasks = 2;
channels = 56;
segment_length = 500; % Data segment length
window = 250; % Window length, ensure less than input signal length
noverlap = 125; % Overlap portion
srate = 1000; % Sampling rate
psd_features = cell(subjects, tasks);
relative_psd_features = cell(subjects, tasks);
% Frequency band definitions
bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
band_ranges = [0.5, 4; 4, 8; 8, 13; 13, 30; 30, 60]; % Each row corresponds to the upper and lower limits of a frequency band
for subj = 1:subjects
for task = 1:tasks
% Load data file
% basePath = 'C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data\\'; 
% basePath = 'C:\\Matlab Working Path\\edata\\ipad-TMT-a-b(.mat_file)_data\\';
% basePath = 'C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\';
        basePath = 'C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ\\';
        filename = sprintf('%ssubj%d_task%d.mat', basePath, subj, task);
        data = load(filename); % Assume data is stored in a variable, for example 'chunks'
        EEGdata = data.chunks; % 163 x 1 cell, each cell size is 56 x 500
% Initialize PSD matrix
        n_segments = length(EEGdata);
% First compute one segment to obtain the size of the frequency information
        [pxx, f] = pwelch(EEGdata{1}(1,:), window, noverlap, [], srate);
        num_freqs = length(pxx);
        psd_matrix_all_segments = zeros(channels, num_freqs, n_segments);
for seg = 1:n_segments
for ch = 1:channels
% Use pwelch to calculate the power spectral density for each segment
                [pxx, f] = pwelch(EEGdata{seg}(ch,:), window, noverlap, [], srate);
                psd_matrix_all_segments(ch, :, seg) = pxx;
end
end
% Average over all segments
        psd_matrix = mean(psd_matrix_all_segments, 3);
        psd_features{subj, task} = psd_matrix;
% Calculate relative PSD
        relative_psd_matrix = zeros(channels, 5); % Assume there are 5 frequency bands: delta, theta, alpha, beta, gamma
        delta_idx = f >= 0.5 & f < 4;
        theta_idx = f >= 4 & f < 8;
        alpha_idx = f >= 8 & f < 13;
        beta_idx = f >= 13 & f < 30;
        gamma_idx = f >= 30 & f < 60;
for ch = 1:channels
            delta_power = sum(psd_matrix(ch, delta_idx));
            theta_power = sum(psd_matrix(ch, theta_idx));
            alpha_power = sum(psd_matrix(ch, alpha_idx));
            beta_power = sum(psd_matrix(ch, beta_idx));
            gamma_power = sum(psd_matrix(ch, gamma_idx));
            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power;
            relative_psd_matrix(ch, 1) = delta_power / total_power;
            relative_psd_matrix(ch, 2) = theta_power / total_power;
            relative_psd_matrix(ch, 3) = alpha_power / total_power;
            relative_psd_matrix(ch, 4) = beta_power / total_power;
            relative_psd_matrix(ch, 5) = gamma_power / total_power;
end
        relative_psd_features{subj, task} = relative_psd_matrix;
end
end