% bands variable definition
bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
% Initialize variable to store relative power spectral density
average_relative_psd_all_subjects = cell(2, 5); % Two types of tasks, five waveforms
% Calculate the average relative power spectral density for each type of task and each waveform
for task = 1:2 % Two types of tasks
for band = 1:5 % Five waveforms
% Initialize accumulator
        average_relative_psd = zeros(56, 1);
        count = 0;
        subjects = 1;
% Traverse each subject, accumulate the relative power spectral density for the same task and same waveform
for subj = 1:subjects % Process each subject in sequence
            relative_psd = relative_psd_features{subj, task}; % Get the relative PSD data for the subj-th subject and task-th type of task
% Extract data for the current frequency band
            band_power = relative_psd(:, band);
% Accumulate relative power spectral density data
            average_relative_psd = average_relative_psd + band_power;
            count = count + 1;
end
% Calculate average, only perform average calculation if count > 0
if count > 0
            average_relative_psd = average_relative_psd / count;
else
            warning('For task %d, no subjects have data in the %s band.', task, bands{band});
end
% Store relative power spectral density
        average_relative_psd_all_subjects{task, band} = average_relative_psd;
end
end
% Load electrode position information
chanlocData = load('C:\\Matlab Working Path\\result\\PC_easy_hard_psd_result\\chanlocs.mat');
if isfield(chanlocData, 'chanlocs')
    chanlocs = chanlocData.chanlocs;
else
    error('The ''chanlocs'' field does not exist in the loaded file. Please check the file structure.');
end
% New code, custom color bar range
% 1. PP range: -0.4-0.4
% 2. tablet range: -0.2-0.6
% 3. VR range: -0.3-0.5
global_min = -0.3; % Set lower limit
global_max = 0.5; % Set upper limit
% topoplot(data, chanlocs, 'maplimits', [global_min global_max], ...);
% Draw brain topography map of average relative power spectral density
figure;
for task = 1:2
for band = 1:5
        subplot(2, 5, (task - 1) * 5 + band);
% topoplot(average_relative_psd_all_subjects{task, band}, chanlocs, 'maplimits', 'absmax', 'style', 'both', 'electrodes', 'off'); % The 'off' parameter specifies not to display electrode names
        topoplot(average_relative_psd_all_subjects{task, band}, chanlocs, 'maplimits', [global_min global_max], 'style', 'both', 'electrodes', 'off');
        title(sprintf('Task %d - %s', task, bands{band}),'FontSize', 20);
end
end
% Add overall title and color bar
% sgtitle('Average Relative PSD across all subjects');
sgtitle('Average PSD across all subjects','FontSize', 25);
% colorbar; % Add color bar
% Add global unified colorbar
h = colorbar('Position', [0.93 0.23 0.02 0.55]);
h.FontSize = 12; % Set font size of color bar ticks
ylabel(h, '', 'FontSize', 20);