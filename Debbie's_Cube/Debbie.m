% Step 1: Extract data from bimodal.dat and bimodal.hdr

% Specify file names
data_file = 'bimodal.dat';
header_file = 'bimodal.hdr';

% Use the provided enviread_new function to read the hyperspectral cube
[data, bands_to_read, data_type, interleave] = enviread_new(data_file);

% Use the provided get_header_param function to extract metadata
[samples, lines, bands, data_type, interleave] = get_header_param(header_file);

% Step 2: Extract data from target location [5, 3]

% Assuming the data is already loaded into the variable `data`
% and dimensions have been validated in Step 1.

% Specify the target location
target_location = [5, 3]; % (row, column)

% Extract the spectral data for all bands at the target location
[row, col] = deal(target_location(1), target_location(2));
target_spectrum = squeeze(data(row, col, :)); % Extract data for all bands

% Display the extracted spectral data
disp('Spectral data at target location [5, 3]:');
disp(target_spectrum);

% Optional: Plot the spectral data for visualization
figure;
plot(target_spectrum);
title('Spectral Data at Target Location [5, 3]');
xlabel('Band Number');
ylabel('Intensity');
grid on;

% Step 3: Modify the data cube by adding the target spectrum

% Assuming `data` is the hyperspectral cube and `target_spectrum` is loaded
% from Step 2. Ensure `data` and `target_spectrum` exist.

% Parameters
p = 0.01; % Target strength
[r, c, b] = size(data); % Dimensions of the data cube
target_spectrum = target_spectrum(:); % Ensure target spectrum is a column vector

% Replicate the target spectrum across the spatial dimensions
target_cube = repmat(reshape(target_spectrum, [1, 1, b]), [r, c, 1]);

% Compute x' = x + p * t
modified_data = data + p * target_cube;

% Step 4: Calculate mean (m) using the 8 closest neighbors

% Assuming `data` is the modified hyperspectral cube from Step 3
[r, c, b] = size(data); % Dimensions of the data cube

% Initialize a matrix to store the mean values
mean_cube = zeros(r, c, b);

% Define neighbor offsets (relative positions)
neighbor_offsets = [
    -1, -1; -1, 0; -1, 1;
     0, -1;        0, 1;
     1, -1;  1, 0;  1, 1
];

% Iterate over each band
for band = 1:b
    % Extract the current band
    current_band = data(:, :, band);
    
    % Compute the mean for each pixel based on neighbors
    for i = 1:r
        for j = 1:c
            % Collect neighbors
            neighbors = [];
            for k = 1:size(neighbor_offsets, 1)
                ni = i + neighbor_offsets(k, 1); % Neighbor row
                nj = j + neighbor_offsets(k, 2); % Neighbor column
                if ni >= 1 && ni <= r && nj >= 1 && nj <= c % Check bounds
                    neighbors = [neighbors; current_band(ni, nj)];
                end
            end
            % Compute the mean of neighbors
            mean_cube(i, j, band) = mean(neighbors);
        end
    end
end

% Step 5: Compute x-m and x'-m for specific locations (First Band Only)

% Specified locations
locations = [5, 5; 10, 10; 15, 15; 20, 20]; 

% Assuming `data` is the original hyperspectral cube
% Assuming `modified_data` is the modified cube from Step 3
% Assuming `mean_cube` is the calculated mean from Step 4

% Initialize arrays to store results
x_minus_m = zeros(size(locations, 1), 1);  % For x-m
x_prime_minus_m = zeros(size(locations, 1), 1);  % For x'-m

% Compute differences for the first band
band = 1; % First band
for i = 1:size(locations, 1)
    loc = locations(i, :); % Current location
    row = loc(1);
    col = loc(2);
    
    % Compute x-m
    x_minus_m(i) = data(row, col, band) - mean_cube(row, col, band);
    
    % Compute x'-m
    x_prime_minus_m(i) = modified_data(row, col, band) - mean_cube(row, col, band);
end

% Display the results in the desired table format
disp('For t^T*phi^-1(x''-m)')
disp('| Pixel Location |    x - m    |   x'' - m   |');
disp('|----------------|-------------|-------------|');
for i = 1:size(locations, 1)
    fprintf('| (%2d, %2d)       | %10.4f | %10.4f |\n', ...
        locations(i, 1), locations(i, 2), x_minus_m(i), x_prime_minus_m(i));
end

% Step 6: Calculate Covariance Matrix Directly

% Assuming `data` is the original hyperspectral data cube
% and `mean_cube` is the mean cube calculated earlier
[rows, cols, bands] = size(data);

% Initialize covariance matrix
covariance_matrix = zeros(bands, bands);

% Total number of pixels
N = rows * cols;

% Loop through all pixels to compute the covariance matrix
for row = 1:rows
    for col = 1:cols
        % Extract the pixel vector and its corresponding mean
        x_vector = squeeze(data(row, col, :)); % Pixel vector (bands x 1)
        m_vector = squeeze(mean_cube(row, col, :)); % Mean vector (bands x 1)

        % Compute (x_i - m) and its outer product
        diff = x_vector - m_vector; % (bands x 1)
        outer_product = diff * diff'; % Outer product (bands x bands)

        % Accumulate the outer products
        covariance_matrix = covariance_matrix + outer_product;
    end
end

% Finalize the covariance matrix by dividing by (N - 1)
covariance_matrix = covariance_matrix / (N - 1);

% Step 7: Compute t^T * Phi^(-1) * (x - m) and t^T * Phi^(-1) * (x' - m) for entire data

% Required inputs:
% `target_spectrum`: Target spectrum vector (bands x 1)
% `data`: Original hyperspectral data cube (rows x cols x bands)
% `modified_data`: Modified data cube (rows x cols x bands)
% `mean_cube`: Mean cube (rows x cols x bands)
% `covariance_matrix`: Covariance matrix (bands x bands)

% Validate covariance matrix and compute its inverse
if det(covariance_matrix) <= 1e-10
    error('Covariance matrix is nearly singular and cannot be reliably inverted.');
end
covariance_matrix_inv = inv(covariance_matrix);

% Get dimensions of the data cube
[rows, cols, bands] = size(data);

% Initialize result matrices
results_x_minus_m = zeros(rows, cols); % Results for x - m
results_x_prime_minus_m = zeros(rows, cols); % Results for x' - m

% Loop through each pixel in the data cube
for row = 1:rows
    for col = 1:cols
        % Compute (x - m) for the current pixel
        x_minus_m = squeeze(data(row, col, :) - mean_cube(row, col, :));
        results_x_minus_m(row, col) = target_spectrum' * covariance_matrix_inv * x_minus_m;
        
        % Compute (x' - m) for the current pixel
        x_prime_minus_m = squeeze(modified_data(row, col, :) - mean_cube(row, col, :));
        results_x_prime_minus_m(row, col) = target_spectrum' * covariance_matrix_inv * x_prime_minus_m;
    end
end

% Display specific location results for verification
locations = [5, 5; 10, 10; 15, 15; 20, 20];
disp('For t^T*phi^-1(x''-m)')
disp('| Pixel Location |    x - m    |   x'' - m   |');
disp('|----------------|-------------|-------------|');
for i = 1:size(locations, 1)
    row = locations(i, 1);
    col = locations(i, 2);
    fprintf('| (%2d, %2d)       | %10.4f | %10.4f |\n', ...
        row, col, results_x_minus_m(row, col), results_x_prime_minus_m(row, col));
end

% Save results for future use
save('results_for_entire_data.mat', 'results_x_minus_m', 'results_x_prime_minus_m');

% Step 8: Plot histograms for NT (x - m) and WT (x' - m) for the entire data

% Assuming `results_x_minus_m` and `results_x_prime_minus_m` are available
% These are the results computed for the entire data cube

% Flatten the matrices into vectors for histogram computation
flattened_NT = results_x_minus_m(:); % Flatten NT (x - m)
flattened_WT = results_x_prime_minus_m(:); % Flatten WT (x' - m)

% Define bins for the histograms
edges = linspace(min([flattened_NT; flattened_WT]), max([flattened_NT; flattened_WT]), 100); % 100 bins

% Compute histograms
counts_NT = histcounts(flattened_NT, edges); % NT histogram
counts_WT = histcounts(flattened_WT, edges); % WT histogram

% Compute bin centers for plotting
bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

% Plot the histograms on the same plot
figure;
hold on;
plot(bin_centers, counts_NT, 'b-', 'LineWidth', 0.8, 'DisplayName', 'NT'); % NT in blue
plot(bin_centers, counts_WT, 'g-', 'LineWidth', 0.8, 'DisplayName', 'WT'); % WT in green
hold off;

% Add labels, legend, and title
xlabel('Result Values'); % X-axis represents result values
ylabel('Number of Pixels'); % Y-axis represents pixel counts
legend('show');
title('Histogram of Results: NT vs WT');
grid on;

% Step 9: Plot ROC curve with x-axis limited to [0, 0.1] and jumps of 0.01

% Assuming `results_x_minus_m` (NT) and `results_x_prime_minus_m` (WT) are available

% Define thresholds for ROC curve computation
thresholds = linspace(min(min(results_x_minus_m(:)), min(results_x_prime_minus_m(:))), ...
                      max(max(results_x_minus_m(:)), max(results_x_prime_minus_m(:))), 100);

% Initialize arrays to store probabilities
P_FA = zeros(length(thresholds), 1); % Probability of False Alarm (P_FA)
P_D = zeros(length(thresholds), 1); % Probability of Detection (P_D)

% Compute P_FA and P_D for each threshold
for i = 1:length(thresholds)
    threshold = thresholds(i);
    
    % P_FA: Probability that NT exceeds the threshold
    P_FA(i) = sum(results_x_minus_m(:) >= threshold) / numel(results_x_minus_m);
    
    % P_D: Probability that WT exceeds the threshold
    P_D(i) = sum(results_x_prime_minus_m(:) >= threshold) / numel(results_x_prime_minus_m);
end

% Plot the ROC curve
figure;
plot(P_FA, P_D, 'b-', 'LineWidth', 1.5, 'DisplayName', 'ROC Curve'); % ROC curve in blue
hold on;

% Plot the diagonal line for P_D = P_FA
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5, 'DisplayName', 'P_D = P_F_A'); % Diagonal line
hold off;

% Add labels, legend, and title
xlabel('P_{FA} (False Alarm Rate)');
ylabel('P_{D} (Detection Probability)');
title('ROC Curve, for th = 0.1');
legend('show');
grid on;

% Adjust axis limits and ticks
xlim([0, 0.1]); % Limit x-axis to [0, 0.1]
ylim([0, 1]); % Y-axis spans [0, 1]
xticks(0:0.01:0.1);
yticks(0:0.1:1); 

% Step 10: Calculate AUC for thresholds 0.001, 0.01, and 0.1

% Define thresholds
thresholds = [0.001, 0.01, 0.1];

% Initialize storage for AUC results
auc_results = zeros(length(thresholds), 1);

% Loop through each threshold to compute AUC
for t_idx = 1:length(thresholds)
    th = thresholds(t_idx);
    
    % Limit the ROC curve to the range [0, th]
    valid_indices = P_FA <= th; % Find indices where P_FA <= th
    P_FA_trimmed = P_FA(valid_indices); % Trim P_FA to the specified range
    P_D_trimmed = P_D(valid_indices); % Trim P_D to the corresponding values
    
    % Ensure that P_FA_trimmed is sorted in ascending order
    [P_FA_trimmed, sort_indices] = sort(P_FA_trimmed);
    P_D_trimmed = P_D_trimmed(sort_indices);
    
    % Calculate the total AUC for the range [0, th] using the trapezoidal rule
    auc_results(t_idx) = trapz(P_FA_trimmed, P_D_trimmed);
end

% Step 11: Calculate A_th for thresholds 0.001, 0.01, and 0.1

% Define thresholds
thresholds = [0.001, 0.01, 0.1];

% Initialize storage for results
results = zeros(length(thresholds), 3); % Columns: [Threshold, Total AUC, Adjusted A_th]

% Loop through each threshold to compute A_th
for t_idx = 1:length(thresholds)
    th = thresholds(t_idx);
    
    % Limit the ROC curve to the range [0, th]
    valid_indices = P_FA <= th; % Find indices where P_FA <= th
    P_FA_trimmed = P_FA(valid_indices); % Trim P_FA to the specified range
    P_D_trimmed = P_D(valid_indices); % Trim P_D to the corresponding values
    
    % Ensure that P_FA_trimmed is sorted in ascending order
    [P_FA_trimmed, sort_indices] = sort(P_FA_trimmed);
    P_D_trimmed = P_D_trimmed(sort_indices);
    
    % Calculate the total AUC for the range [0, th] using the trapezoidal rule
    auc_total = trapz(P_FA_trimmed, P_D_trimmed);
    
    % Compute the triangular area under P_D = P_FA for the threshold
    triangle_area = 0.5 * th^2; % Triangle area
    
    % Compute adjusted A_th using the formula
    adjusted_auc_th = (auc_total - triangle_area) / (th - triangle_area);
    
    % Store the results
    results(t_idx, :) = [th, auc_total, adjusted_auc_th];
end

% Display the results
disp('Threshold   Total AUC   Adjusted A_th');
for i = 1:size(results, 1)
    fprintf('%.4f       %.4f       %.4f\n', results(i, 1), results(i, 2), results(i, 3));
end


















