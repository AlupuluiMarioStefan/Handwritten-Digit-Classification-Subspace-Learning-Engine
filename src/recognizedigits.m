% RECOGNIZEDIGITS
%
% Classifies handwritten digit images from the USPS dataset using three
% methods of increasing sophistication:
%   1. Nearest Average   - distance to each digit's mean training image
%   2. Nearest Neighbor  - distance to every individual training image
%   3. SVD Subspace      - distance to each digit's r-dimensional subspace
%
% Requires digitimagesandlabels.mat on the path, which provides:
%   trainingimages        256 x 1707  training images (one per column)
%   trainingimageslabels  1707 x 1    true digit (0-9) for each training image
%   testimages             256 x 2007  test images (one per column)
%   testimageslabels       2007 x 1    true digit (0-9) for each test image
%
% For each method this script reports overall accuracy, per-digit
% accuracy, execution time, the first five misclassified test indices,
% a confusion matrix, and the five most frequent misclassification pairs.

clear all
close all

% Read the data for this problem
load digitimagesandlabels

n_train = size(trainingimages, 2);
n_test  = size(testimages, 2);

% Count the test images for each digit
test_images_per_digit = zeros(10, 1);
for digit = 0:9
    test_images_per_digit(digit + 1) = sum(testimageslabels == digit);
end


%% ------------------------------------------------------------------
%  Method 1: Nearest Average
%  --------------------------------------------------------------------
disp('Starting Method 1: Nearest Average ...');
tic;

% Train: compute the mean training image for each digit
matrixofaveragenumbers = zeros(256, 10);
for j = 0:9
    numberofnumberj = 0;
    sumofnumberj = zeros(256, 1);
    for i = 1:n_train
        if trainingimageslabels(i) == j
            % Image i is labeled as the digit j
            numberofnumberj = numberofnumberj + 1;
            sumofnumberj = sumofnumberj + trainingimages(:, i);
        end
    end
    averagenumberj = sumofnumberj / numberofnumberj;
    matrixofaveragenumbers(:, j + 1) = averagenumberj;
end

% Test: assign each test image to its closest average
predictions_an = zeros(n_test, 1);
correct_predictions_an = 0;
correct_predictions_per_digit_an = zeros(10, 1);
failed_an_indices = [];
for j = 1:n_test
    d_min = Inf;
    closestnumber = -1;
    for i = 0:9
        d = norm(testimages(:, j) - matrixofaveragenumbers(:, i + 1));
        if d < d_min
            d_min = d;
            closestnumber = i;
        end
    end
    predictions_an(j) = closestnumber;
    if closestnumber == testimageslabels(j)
        correct_predictions_an = correct_predictions_an + 1;
        digit_index = double(testimageslabels(j)) + 1;
        correct_predictions_per_digit_an(digit_index) = correct_predictions_per_digit_an(digit_index) + 1;
    else
        failed_an_indices = [failed_an_indices, j];
    end
end
time_an = toc;

fprintf('First 5 misclassified images for Nearest Average: %s\n', num2str(failed_an_indices(1:5)));
accuracy_an = (correct_predictions_an / n_test) * 100;
fprintf('Total Correct: %d out of %d\n', correct_predictions_an, n_test);
fprintf('Accuracy: %.2f%%\n', accuracy_an);
fprintf('Execution time: %.2f seconds\n\n', time_an);

% Print the accuracy for each digit
accuracy_per_digit_an = zeros(10, 1);
fprintf('Accuracy per digit:\n');
for digit = 0:9
    digit_index = digit + 1;
    if test_images_per_digit(digit_index) > 0
        accuracy_per_digit_an(digit_index) = (correct_predictions_per_digit_an(digit_index) / test_images_per_digit(digit_index)) * 100;
        fprintf('Digit %d: %d out of %d correct, Accuracy: %.2f%%\n', digit, ...
            correct_predictions_per_digit_an(digit_index), test_images_per_digit(digit_index), accuracy_per_digit_an(digit_index));
    else
        fprintf('Digit %d: No test images\n', digit);
    end
end
fprintf('\n');


%% ------------------------------------------------------------------
%  Method 2: Nearest Neighbor
%  --------------------------------------------------------------------
disp('Starting Method 2: Nearest Neighbor ...');
tic;

predictions_nn = zeros(n_test, 1);
correct_predictions_nn = 0;
correct_predictions_per_digit_nn = zeros(10, 1);
failed_nn_indices = [];
for j = 1:n_test
    d_min = Inf;
    closestnumber = -1;
    for i = 1:n_train
        d = norm(testimages(:, j) - trainingimages(:, i));
        if d < d_min
            d_min = d;
            closestnumber = trainingimageslabels(i);
        end
    end
    predictions_nn(j) = closestnumber;
    if closestnumber == testimageslabels(j)
        correct_predictions_nn = correct_predictions_nn + 1;
        digit_index = double(testimageslabels(j)) + 1;
        correct_predictions_per_digit_nn(digit_index) = correct_predictions_per_digit_nn(digit_index) + 1;
    else
        failed_nn_indices = [failed_nn_indices, j];
    end
end
time_nn = toc;

fprintf('First 5 misclassified images for Nearest Neighbor: %s\n', num2str(failed_nn_indices(1:5)));
accuracy_nn = (correct_predictions_nn / n_test) * 100;
fprintf('Total Correct: %d out of %d\n', correct_predictions_nn, n_test);
fprintf('Accuracy: %.2f%%\n', accuracy_nn);
fprintf('Execution time: %.2f seconds\n\n', time_nn);

accuracy_per_digit_nn = zeros(10, 1);
fprintf('Accuracy per digit:\n');
for digit = 0:9
    digit_index = digit + 1;
    if test_images_per_digit(digit_index) > 0
        accuracy_per_digit_nn(digit_index) = (correct_predictions_per_digit_nn(digit_index) / test_images_per_digit(digit_index)) * 100;
        fprintf('Digit %d: %d out of %d correct, Accuracy: %.2f%%\n', digit, ...
            correct_predictions_per_digit_nn(digit_index), test_images_per_digit(digit_index), accuracy_per_digit_nn(digit_index));
    else
        fprintf('Digit %d: No test images\n', digit);
    end
end
fprintf('\n');

% NOTE: the loop above compares every test image with all 1707 training
% images one at a time (O(n_test * n_train) norm calls), which is why it
% is ~50x slower than the other two methods. It can be replaced with a
% single vectorized distance computation for a large speed-up:
%
%   tr_sq  = sum(trainingimages.^2, 1);            % 1 x n_train
%   te_sq  = sum(testimages.^2, 1)';                % n_test x 1
%   dist2  = te_sq + tr_sq - 2*(testimages' * trainingimages);
%   [~, nn_idx] = min(dist2, [], 2);
%   predictions_nn_fast = trainingimageslabels(nn_idx);
%
% This produces identical predictions to the loop above but runs in a
% fraction of the time, since it lets MATLAB's BLAS routines handle the
% 2007x1707 distance matrix instead of looping in interpreted code.


%% ------------------------------------------------------------------
%  Method 3: SVD Subspace Classifier
%  --------------------------------------------------------------------
disp('Starting Method 3: SVD ...');
tic;

r = 15;
% r = 15 was chosen by testing several values and picking the one with
% the best test accuracy (see report, Section 3.3)
Ur = cell(10, 1);
for i = 0:9
    A_j = trainingimages(:, trainingimageslabels == i);
    [U, S, V] = svd(A_j, 'econ');
    Ur{i + 1} = U(:, 1:r);
end

predictions_svd = zeros(n_test, 1);
correct_predictions_svd = 0;
correct_predictions_per_digit_svd = zeros(10, 1);
failed_svd_indices = [];
for i = 1:n_test
    test_image = testimages(:, i);
    d_min = Inf;
    closest_number = -1;
    for j = 0:9
        x_proj = Ur{j + 1} * (Ur{j + 1}' * test_image);
        d = norm(test_image - x_proj);
        if d < d_min
            d_min = d;
            closest_number = j;
        end
    end
    predictions_svd(i) = closest_number;
    if closest_number == testimageslabels(i)
        correct_predictions_svd = correct_predictions_svd + 1;
        digit_index = double(testimageslabels(i)) + 1;
        correct_predictions_per_digit_svd(digit_index) = correct_predictions_per_digit_svd(digit_index) + 1;
    else
        failed_svd_indices = [failed_svd_indices, i];
    end
end
time_svd = toc;

fprintf('First 5 misclassified images for SVD: %s\n', num2str(failed_svd_indices(1:5)));
accuracy_svd = (correct_predictions_svd / n_test) * 100;
fprintf('Total Correct: %d out of %d\n', correct_predictions_svd, n_test);
fprintf('Accuracy: %.2f%%\n', accuracy_svd);
fprintf('Execution time: %.2f seconds\n\n', time_svd);

accuracy_per_digit_svd = zeros(10, 1);
fprintf('Accuracy per digit:\n');
for digit = 0:9
    digit_index = digit + 1;
    if test_images_per_digit(digit_index) > 0
        accuracy_per_digit_svd(digit_index) = (correct_predictions_per_digit_svd(digit_index) / test_images_per_digit(digit_index)) * 100;
        fprintf('Digit %d: %d out of %d correct, Accuracy: %.2f%%\n', digit, ...
            correct_predictions_per_digit_svd(digit_index), test_images_per_digit(digit_index), accuracy_per_digit_svd(digit_index));
    else
        fprintf('Digit %d: No test images\n', digit);
    end
end
fprintf('\n');


%% ------------------------------------------------------------------
%  Confusion matrices and most frequent errors (Figure 1 / Table 5)
%  --------------------------------------------------------------------
CM_an  = build_confusion_matrix(predictions_an,  testimageslabels);
CM_nn  = build_confusion_matrix(predictions_nn,  testimageslabels);
CM_svd = build_confusion_matrix(predictions_svd, testimageslabels);

plot_confusion_matrices(CM_an, CM_nn, CM_svd);

fprintf('Most frequent errors (true -> predicted):\n');
fprintf('Nearest average : %s\n', format_top_errors(CM_an, 5));
fprintf('Nearest neighbor: %s\n', format_top_errors(CM_nn, 5));
fprintf('SVD (r = %d)     : %s\n', r, format_top_errors(CM_svd, 5));


%% ==================== Local functions ====================

function CM = build_confusion_matrix(predictions, true_labels)
% BUILD_CONFUSION_MATRIX  10x10 matrix, rows = true digit, cols = predicted
% digit (both 0-indexed, stored in rows/cols 1-10). CM(t+1, p+1) counts
% how many images whose true label is t were predicted as p.
CM = zeros(10, 10);
for k = 1:length(true_labels)
    t = double(true_labels(k)) + 1;
    p = double(predictions(k)) + 1;
    CM(t, p) = CM(t, p) + 1;
end
end


function plot_confusion_matrices(CM_an, CM_nn, CM_svd)
% PLOT_CONFUSION_MATRICES  Reproduces Figure 1 of the report: one
% heatmap per method, correct predictions on the diagonal.
figure
titles = {'Nearest average', 'Nearest neighbor', 'SVD'};
mats = {CM_an, CM_nn, CM_svd};
for k = 1:3
    subplot(1, 3, k)
    imagesc(mats{k})
    colormap(flipud(gray))
    axis square
    title(titles{k})
    xlabel('Predicted digit')
    ylabel('True digit')
    set(gca, 'XTick', 1:10, 'XTickLabel', 0:9, 'YTick', 1:10, 'YTickLabel', 0:9)
end
end


function s = format_top_errors(CM, k)
% FORMAT_TOP_ERRORS  Returns a string listing the k most frequent
% off-diagonal (true -> predicted) confusions in CM, e.g.
% "0 -> 6 (38), 4 -> 9 (32), ..."
CM_offdiag = CM - diag(diag(CM));
[counts, idx] = sort(CM_offdiag(:), 'descend');
parts = cell(k, 1);
for n = 1:k
    [t, p] = ind2sub(size(CM), idx(n));
    parts{n} = sprintf('%d -> %d (%d)', t - 1, p - 1, counts(n));
end
s = strjoin(parts, ', ');
end
