%% Plot 3D Wavelet Transform Figure
% clear; clc; close all;
numSubjects = 8; % Number of subjects
numTasks = 2; % Number of tasks per subject
numChannels = 56; % Number of EEG channels
scales = 1:30; % Scales for wavelet transform
wname = 'morl'; % Wavelet name, using Morlet wavelet
downsample_factor = 10; % Time dimension downsampling factor (1/10)
% Add save path variable (modifiable)
savePath = 'D:\1.EEG_Fig\WT_3D'; % Save path
if ~exist(savePath, 'dir'), mkdir(savePath); end
% Define electrode indices representing different brain regions
% 1. PP-TMT
electrodeIndices = struct('FPz', 2, 'F7', 6, 'FP2', 3, 'T8', 30, 'T7',22,'FP1',1);
% 2. Tablet-TMT (ipad)
% electrodeIndices = struct('FPz', 2, 'FP2', 3, 'AF4', 5, 'FP1', 1, 'F4',12,'F6',13);
% 3. VR-TMT
% electrodeIndices = struct('F8', 14, 'FPz', 2, 'FP2', 3, 'FP1', 1, 'F7',6,'F6',13);
fields = fieldnames(electrodeIndices);
nElec = numel(fields);
% Pre-allocate plotting data structure
totalCwtCoeffs_plot = struct();
for i = 1:nElec
    totalCwtCoeffs_plot.(fields{i}) = [];
end
% Pre-generate file list to improve I/O efficiency
fileList = cell(numSubjects, numTasks);
for subj = 1:numSubjects
for task = 1:numTasks
% Load EEG data path
        basePath = 'C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\';
% basePath = 'C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data\\';
% basePath = 'C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ\\';
        fileList{subj, task} = sprintf('%ssubj%d_task%d.mat', basePath, subj, task);
end
end
% Serial loop
for subj = 1:numSubjects
for task = 1:numTasks
        filename = fileList{subj, task};
if ~exist(filename, 'file')
            warning('File not found: %s', filename);
continue;
end
        data = load(filename);
        chunks = data.chunks;
% Process each chunk
for chunkIdx = 1:length(chunks)
            chunk = chunks{chunkIdx};
            nSamples = size(chunk, 2); % Get number of samples
% WT for each electrode
for i = 1:nElec
                field = fields{i};
                ch = electrodeIndices.(field);
                cwtCoeffs = cwt(chunk(ch, :), scales, wname);
                coeffs_abs = abs(cwtCoeffs); % Absolute magnitude
% Downsample time dimension (columns)
                coeffs_ds = coeffs_abs(:, 1:downsample_factor:end);
% Accumulate downsampled data
                totalCwtCoeffs_plot.(field) = [totalCwtCoeffs_plot.(field), coeffs_ds];
end
end
end
end
% Calculate global min/max for clim
global_min = inf;
global_max = -inf;
for i = 1:nElec
    data = totalCwtCoeffs_plot.(fields{i});
    global_min = min(global_min, min(data(:)));
    global_max = max(global_max, max(data(:)));
end
fprintf('Global magnitude range: [%.4f, %.4f]\n', global_min, global_max);
% Plot 3D surface figure
figure('Position', [100, 100, 1800, 900]); % Set figure size
for i = 1:nElec
    field = fields{i};
    subplot(2, 3, i);
    data = totalCwtCoeffs_plot.(field); % Downsampled data
    [X, Y] = meshgrid(1:size(data,2), scales);
    surf(X, Y, data, 'EdgeColor', 'none');
    xlabel('Time','FontSize',20);
    ylabel('Scale','FontSize',20);
    zlabel('Magnitude of wavelet coefficients','FontSize',16);
% title(sprintf('%s 3D plot of WT time-frequency magnitude', field), 'FontSize', 18);
    title(sprintf('%s', field), 'FontSize', 18);
    set(gca, 'FontSize', 16);
    view(45, 30); shading interp;
% Custom clim (modify here as needed)
% clim([global_min, global_max]); % Or PP: clim([0,300])
% clim([global_min, global_max]); % Or clim([0,450])
% clim([global_min, global_max]); % Or clim([0,700]) etc.
    clim([0,500])
end
% Add overall title (placed above all subplots)
sgtitle('PP-TMT 3D plot of WT time-frequency magnitude', ...
'FontSize', 20, 'FontWeight', 'bold');
% sgtitle('Tablet-TMT 3D plot of WT time-frequency magnitude', ...
% 'FontSize', 20, 'FontWeight', 'bold');
% sgtitle('VR-TMT Representative Electrodes Time–Frequency Maps', ...
% 'FontSize', 20, 'FontWeight', 'bold');
% Unified colorbar
h = colorbar('Position', [0.93 0.23 0.02 0.55]);
h.FontSize = 16;
ylabel(h, 'Magnitude of wavelet coefficients', 'FontSize', 20);
% % Save interactive figure .fig
% savefig(gcf, fullfile(savePath, 'WT_3D_PP.fig'));
% savefig(gcf, fullfile(savePath, 'WT_3D_Tablet.fig'));
% savefig(gcf, fullfile(savePath, 'WT_3D_VR.fig'));
% fprintf('Saved .fig file size: %.2f MB\n', dir(fullfile(savePath, 'WT_3D_VR2.fig')).bytes / 1e6);