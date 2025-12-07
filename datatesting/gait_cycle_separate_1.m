%% Phase-Space Loops For Each Detected Step
clear; clc; close all;

%% === Load your data (REPLACE WITH YOUR FILE NAMES) ===
angles = readtable('BT10_stairs_2_5_up_off_angle_filt.csv');
vels   = readtable('BT10_stairs_2_5_up_off_velocity.csv');

%% === Sync the tables using time ===
angles = sortrows(angles,"time");
vels   = sortrows(vels,"time");

data.time = angles.time;
data.hip = angles.hip_flexion_l;
data.hip_vel = interp1(vels.time, vels.hip_flexion_l_velocity, angles.time, 'linear');

t = data.time;
theta = data.hip;
theta_vel = data.hip_vel;

%% === Detect steps ===
mean_v = mean(theta_vel);
std_v = std(theta_vel);
thresh = mean_v + std_v;

[peakIdx, ~] = findpeaks(theta_vel, 'MinPeakHeight', thresh);

% ---- 修正 index，避免 colon 运算报错 ----
peakIdx = round(peakIdx);
peakIdx = peakIdx(peakIdx >= 1);
peakIdx = peakIdx(peakIdx <= length(theta));

if length(peakIdx) < 2
    error('Not enough steps detected.');
end

%% === For each detected step, compute phase variables ===
figure; hold on;

for k = 1:length(peakIdx)-1

    i0 = peakIdx(k);
    i1 = peakIdx(k+1);

    th = theta(i0:i1);
    tt = t(i0:i1);

    dt = mean(diff(tt));
    th_mean = mean(th);

    % Integral of (θ − mean θ)
    y = cumsum((th - th_mean) * dt);

    % Plot loop
    plot(th, y, 'LineWidth', 2);
end

%% === Final figure formatting ===
title('Phase-Space Loops For Each Detected Step');
xlabel('Hip Flexion (deg)');
ylabel('Integral of (\theta - mean \theta)');
grid on;
set(gca, 'FontSize', 14);
legend("Step 1", "Step 2");
disp(peakIdx);

