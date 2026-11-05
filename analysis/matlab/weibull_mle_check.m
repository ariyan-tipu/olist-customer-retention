% Cross-check of R's Weibull AFT survival fit by direct maximum likelihood.
%
% Purpose: reproduce the survreg() result from the likelihood itself.
% The censored Weibull log-likelihood is written out from first principles
% and maximised numerically with fminsearch; the estimates must agree with
% R's Newton-Raphson fit to tight tolerances or this script fails.
%
% Parameterisation (matching survreg): log(T) = X*beta + sigma*W, where W
% follows the standard smallest extreme value distribution. With
% z = (log(t) - X*beta) / sigma, each observation contributes
%
%   event:    log f(t) = z - exp(z) - log(sigma) - log(t)
%   censored: log S(t) =   - exp(z)
%
% Censored customers therefore still inform the fit through their
% survival term; dropping them is the mistake this project keeps refusing
% to make.
%
% Inputs (written by analysis/r/survival_analysis.R):
%   outputs/survival/aft_inputs.csv    columns: time_days, event,
%                                      late_delivery, low_review
%   outputs/survival/aft_params_r.csv  R estimates: intercept,
%                                      coef_late_delivery, coef_low_review,
%                                      log_scale, loglik, n, events
%
% Runs unchanged in MATLAB (R2025b, via matlab -batch) and GNU Octave
% (CI), using only core functions. Set OLIST_SURVIVAL_DIR to override the
% input directory.

% run() executes this script from inside analysis/matlab, so the default
% input location is resolved from the script's own position rather than
% the caller's working directory. Works identically in MATLAB and Octave.
survival_dir = getenv('OLIST_SURVIVAL_DIR');
if isempty(survival_dir)
    script_dir = fileparts(mfilename('fullpath'));
    survival_dir = fullfile(script_dir, '..', '..', 'outputs', 'survival');
end

inputs_path = fullfile(survival_dir, 'aft_inputs.csv');
params_path = fullfile(survival_dir, 'aft_params_r.csv');
if exist(inputs_path, 'file') ~= 2 || exist(params_path, 'file') ~= 2
    error('weibull_mle_check:missingInputs', ...
        'Expected %s and %s. Run the Spark pipeline and the R script first.', ...
        inputs_path, params_path);
end

% Both files are numeric CSVs with one header row.
inputs = dlmread(inputs_path, ',', 1, 0);
r_params = dlmread(params_path, ',', 1, 0);

t = inputs(:, 1);
event = inputs(:, 2);
X = [ones(size(t)), inputs(:, 3), inputs(:, 4)];

r_beta = r_params(1, 1:3)';
r_log_scale = r_params(1, 4);
r_loglik = r_params(1, 5);

if any(t <= 0)
    error('weibull_mle_check:badData', 'Durations must be positive.');
end

fprintf('Loaded %d spells (%d events, %.1f%% censored)\n', ...
    numel(t), sum(event), 100 * (1 - mean(event)));

% Closed-form exponential MLE as an order-of-magnitude sanity anchor:
% with censoring, lambda_hat = events / total exposure.
lambda_hat = sum(event) / sum(t);
fprintf('Exponential sanity check: lambda = %.6g per day (mean %.1f days)\n', ...
    lambda_hat, 1 / lambda_hat);

% Direct maximisation of the censored Weibull log-likelihood. The
% likelihood stays anonymous rather than a script-local function because
% Octave cannot resolve a script's trailing function from inside an
% fminsearch handle. With beta = p(1:3) and log(sigma) = p(4), the
% z values are clamped so exp(z) cannot overflow in early simplex steps.
z_of = @(p) min((log(t) - X * p(1:3)) / exp(p(4)), 50);
neg_loglik = @(p) -sum( ...
    event .* (z_of(p) - exp(z_of(p)) - p(4) - log(t)) ...
    + (1 - event) .* (-exp(z_of(p))));
start = [log(median(t)); 0; 0; 0];
options = optimset('TolX', 1e-10, 'TolFun', 1e-10, ...
    'MaxFunEvals', 100000, 'MaxIter', 100000);
p_hat = fminsearch(neg_loglik, start, options);
% One restart from the optimum guards against a premature simplex stop.
p_hat = fminsearch(neg_loglik, p_hat, options);
matlab_loglik = -neg_loglik(p_hat);

names = {'intercept'; 'coef_late_delivery'; 'coef_low_review'; 'log_scale'};
r_values = [r_beta; r_log_scale];
differences = abs(p_hat - r_values);

fprintf('\n%-22s %14s %14s %12s\n', 'parameter', 'R survreg', 'direct MLE', 'abs diff');
for i = 1:numel(names)
    fprintf('%-22s %14.6f %14.6f %12.2e\n', ...
        names{i}, r_values(i), p_hat(i), differences(i));
end
fprintf('%-22s %14.3f %14.3f %12.2e\n', 'log-likelihood', ...
    r_loglik, matlab_loglik, abs(matlab_loglik - r_loglik));

tol_params = 1e-2;
tol_loglik = 1.0;
if any(differences > tol_params) || abs(matlab_loglik - r_loglik) > tol_loglik
    error('weibull_mle_check:mismatch', ...
        'Direct MLE disagrees with R survreg beyond tolerance.');
end

% Hazard ratio interpretation under Weibull AFT: HR = exp(-coef / sigma).
sigma_hat = exp(p_hat(4));
fprintf('\nImplied hazard ratios (AFT coefficients mapped through the shape):\n');
fprintf('  late delivery: %.3f\n', exp(-p_hat(2) / sigma_hat));
fprintf('  low review:    %.3f\n', exp(-p_hat(3) / sigma_hat));
fprintf('\nCROSS-CHECK PASSED\n');
