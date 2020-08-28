infection_file_path = '../../data/link_prediction/result/prediction_infection_mat_MLP.csv';
PPI_file_path = '../../data/link_prediction/result/prediction_PPI_mat_MLP.csv';

infection_mat = readmatrix(infection_file_path);
PPI_mat = readmatrix(PPI_file_path);

% extract confidences as input
infection_confidences = log(infection_mat(:,2));
PPI_confidences = log(PPI_mat(:,2));

histogram(infection_confidences, 1000);
histogram(PPI_confidences, 1000);

% create distribution based on confidence
infection_dist = fitdist(infection_confidences, 'Normal');
PPI_dist = fitdist(PPI_confidences, 'Normal');

histfit(PPI_confidences, 1000, 'Normal');

% generate p-values
infection_pvals = abs(cdf(infection_dist, infection_confidences) - 1);
PPI_pvals = abs(cdf(PPI_dist, PPI_confidences) - 1);

% extract q-values from p-values using storey's pFDR procedure
[infection_fdr, infection_qvals] = mafdr(infection_pvals);
[PPI_fdr, PPI_qvals] = mafdr(PPI_pvals);

% add p-values, q-values column to the original matrix
infection_mat = [infection_mat infection_pvals infection_qvals];
PPI_mat = [PPI_mat PPI_pvals PPI_qvals];

csvwrite('../../data/link_prediction/result/prediction_infection_adjusted_HalfNorm_Dist_MLP.csv',infection_mat)
csvwrite('../../data/link_prediction/result/prediction_PPI_adjusted_HalfNorm_Dist_MLP.csv',PPI_mat)