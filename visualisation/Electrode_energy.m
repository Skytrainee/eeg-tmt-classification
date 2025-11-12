% Calculate the top N electrodes with the highest energy
%% Define electrode names (using standard electrode names)
electrode_names = {'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', ...
'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', ...
'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', ...
'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', ...
'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', ...
'PO4', 'PO8', 'O1', 'OZ', 'O2'};
% Check if the number of electrode names is 56
if length(electrode_names) ~= 56
    error('Number of electrode names does not match. Expected 56 electrode names.');
end
%% Load combined data for eight subjects for Task 1 and Task 2
% Modify the following paths according to your actual data paths
data_path_task1 = 'C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ-combined\\allTask1Chunks.mat';
data_path_task2 = 'C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ-combined\\allTask2Chunks.mat';
% data_path_task1 = 'C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ-combined\\allTask1Chunks.mat';
% data_path_task2 = 'C:\\Matlab Working Path\\edata\\VR_TMT_data\\0.5-60HZ-combined\\allTask2Chunks.mat';
% data_path_task1 = 'C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data_combined\\allTask1Chunks.mat';
% data_path_task2 = 'C:\\Matlab Working Path\\edata\\ipad-TMT-re-filter(.mat_file)_data_combined\\allTask2Chunks.mat';
% % II. PP-TMT
%
% % load('C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\subj8_task1.mat');
% % chunks_task1 = chunks;
% % load('C:\\Matlab Working Path\\edata\\pp_TMT_data\\0.5-60HZ\\subj8_task2.mat');
% % chunks_task2 = chunks;
%
% % Load combined data for eight subjects for Task 1 and Task 2
% load('C:\Matlab Working Path\edata\pp_TMT_data\0.5-60HZ-combined\\allTask1Chunks.mat');
% chunks_task1 = allTask1Chunks;
% load('C:\Matlab Working Path\edata\pp_TMT_data\0.5-60HZ-combined\\allTask2Chunks.mat');
% chunks_task2 = allTask2Chunks;
% Load Task 1 data
if exist(data_path_task1, 'file')
    load(data_path_task1); % Assume variable name is allTask1Chunks
    chunks_task1 = allTask1Chunks;
else
    error('Task 1 data file does not exist. Please check the path: %s', data_path_task1);
end
% Load Task 2 data
if exist(data_path_task2, 'file')
    load(data_path_task2); % Assume variable name is allTask2Chunks
    chunks_task2 = allTask2Chunks;
else
    error('Task 2 data file does not exist. Please check the path: %s', data_path_task2);
end
%% Initialize energy matrices
num_electrodes = size(chunks_task1{1}, 1); % Should be 56
if num_electrodes ~= 56
    error('Number of electrodes does not match. Expected 56 electrodes, but found %d electrodes.', num_electrodes);
end
energy_task1 = zeros(num_electrodes, length(chunks_task1));
energy_task2 = zeros(num_electrodes, length(chunks_task2));
%% Calculate energy for each chunk in Task 1
for i = 1:length(chunks_task1)
for j = 1:num_electrodes
        energy_task1(j, i) = sum(chunks_task1{i}(j, :).^2);
end
end
%% Calculate energy for each chunk in Task 2
for m = 1:length(chunks_task2)
for n = 1:num_electrodes
        energy_task2(n, m) = sum(chunks_task2{m}(n, :).^2);
end
end
% %% Calculate average energy for each electrode
mean_energy_task1 = mean(energy_task1, 2); % Calculate row mean 56×466
mean_energy_task2 = mean(energy_task2, 2); % Calculate row mean 56×824
%% Calculate energy difference
energy_diff = abs(mean_energy_task1 - mean_energy_task2);
%% Calculate total average energy (sort by total energy)
energy_total = mean_energy_task1 + mean_energy_task2;
%% Sort by total energy in descending order (modified: from energy_diff to mean_energy_total)
[sorted_energy, sort_idx] = sort(energy_total, 'descend');
sorted_electrode_names = electrode_names(sort_idx);
sorted_mean_energy_task1 = mean_energy_task1(sort_idx);
sorted_mean_energy_task2 = mean_energy_task2(sort_idx);
sorted_energy_diff = energy_diff(sort_idx); % Sort differences if needed
%% Select top N electrodes (N is adjustable)
top_n = 20; % Modify this value for the desired number of top electrodes
top_sorted_electrode_names = sorted_electrode_names(1:top_n);
top_mean_energy_task1 = sorted_mean_energy_task1(1:top_n);
top_mean_energy_task2 = sorted_mean_energy_task2(1:top_n);
top_energy_diff = sorted_energy_diff(1:top_n);
%% Visualize energy (combined into a grouped bar chart)
figure('Position', [100, 100, 1200, 600]); % Set figure window size
% Set standard RGB colors and normalize
color_rgb1 = [165, 222, 236] / 255; % Blue for Task 1
color_rgb2 = [243, 165, 152] / 255; % Red for Task 2
color_rgb3 = [171, 132, 182] / 255; % Purple for Difference
% Create data matrix for grouped bar chart
data = [top_mean_energy_task1, top_mean_energy_task2, top_energy_diff];
% Grouped bar chart, and get handle to explicitly set colors
b = bar(data, 'grouped');
% Explicitly set colors for each group to match original
set(b(1), 'FaceColor', color_rgb1);
set(b(2), 'FaceColor', color_rgb2);
set(b(3), 'FaceColor', color_rgb3);
% Set axis labels and title
% xlabel('Electrode', 'FontSize', 18);
xlabel('Electrode', 'FontSize', 18);
ylabel('Energy / Difference', 'FontSize', 18);
title('Average Energy and Difference for Top 20 Electrodes (PP-TMT)', 'FontSize', 20);
% title('Average Energy and Difference for Each Electrodes (PP-TMT)', 'FontSize', 20);
% title('Average Energy and Difference for Top 20 Electrodes (Tablet-TMT)', 'FontSize', 20);
% title('Average Energy and Difference for Each Electrodes (Tablet-TMT)', 'FontSize', 20);
% title('Average Energy and Difference for Top 20 Electrodes (VR-TMT)', 'FontSize', 20);
% title('Average Energy and Difference for Each Electrodes (VR-TMT)', 'FontSize', 20);
% Top 20 electrodes, X-axis tick labels not rotated
set(gca, 'FontSize', 16, 'XTick', 1:top_n, ...
'XTickLabel', top_sorted_electrode_names, 'XTickLabelRotation', 0);
% % All 56 electrodes, X-axis tick labels rotated 90 degrees
% set(gca, 'FontSize', 16, 'XTick', 1:top_n, ...
% 'XTickLabel', top_sorted_electrode_names, 'XTickLabelRotation', 90);
legend('Task 1 Energy', 'Task 2 Energy', 'Energy Difference', 'Location', 'Best');
grid on;
% Save figure
% savePath = 'D:\1.EEG_Fig\Electrodes_Energy_Difference'; % Save path
% savefig(gcf, fullfile(savePath, 'PP-TMT_Electrodes_Energy_Difference_20.fig'));
% savefig(gcf, fullfile(savePath, 'PP-TMT_Electrodes_Energy_Difference_56.fig'));
% savefig(gcf, fullfile(savePath, 'Tablet-TMT_Electrodes_Energy_Difference_20.fig'));
% savefig(gcf, fullfile(savePath, 'Tablet-TMT_Electrodes_Energy_Difference_56.fig'));
% savefig(gcf, fullfile(savePath, 'VR-TMT_Electrodes_Energy_Difference_20.fig'));
% savefig(gcf, fullfile(savePath, 'VR-TMT_Electrodes_Energy_Difference_56.fig'));