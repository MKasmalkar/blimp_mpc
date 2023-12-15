clc
close all

cs_lp_20_err = error_norm(cs_lp_n20);
cs_lp_40_err = error_norm(cs_lp_n40);
cs_lp_60_err = error_norm(cs_lp_n60);
cs_lp_80_err = error_norm(cs_lp_n80);
cs_lp_100_err = error_norm(cs_lp_n100);

cs_lp_errs = [cs_lp_20_err
              cs_lp_40_err
              cs_lp_60_err
              cs_lp_80_err
              cs_lp_100_err];

cs_nlp_20_err = error_norm(cs_nlp_n20);
cs_nlp_40_err = error_norm(cs_nlp_n40);
cs_nlp_60_err = error_norm(cs_nlp_n60);
cs_nlp_80_err = error_norm(cs_nlp_n80);
cs_nlp_100_err = error_norm(cs_nlp_n100);

cs_nlp_errs = [cs_nlp_20_err
               cs_nlp_40_err
               cs_nlp_60_err
               cs_nlp_80_err
               cs_nlp_100_err];

cs_lp_20_st = mean(cs_lp_n20(:, 34));
cs_lp_40_st = mean(cs_lp_n40(:, 34));
cs_lp_60_st = mean(cs_lp_n60(:, 34));
cs_lp_80_st = mean(cs_lp_n80(:, 34));
cs_lp_100_st = mean(cs_lp_n100(:, 34));

cs_lp_sts = [cs_lp_20_st
             cs_lp_40_st
             cs_lp_60_st
             cs_lp_80_st
             cs_lp_100_st];

cs_nlp_20_st = mean(cs_nlp_n20(:, 34));
cs_nlp_40_st = mean(cs_nlp_n40(:, 34));
cs_nlp_60_st = mean(cs_nlp_n60(:, 34));
cs_nlp_80_st = mean(cs_nlp_n80(:, 34));
cs_nlp_100_st = mean(cs_nlp_n100(:, 34));

cs_nlp_sts = [cs_nlp_20_st
              cs_nlp_40_st
              cs_nlp_60_st
              cs_nlp_80_st
              cs_nlp_100_st];

n = [20, 40, 60, 80, 100];

subplot(211)
bar(n, [cs_lp_errs, cs_nlp_errs]);
title("Mean position error")
xlabel('Time horizon (time steps)')
ylabel('Mean error (m)')
legend('Linear MPC', 'Nonlinear MPC')

subplot(212)
bar(n, [cs_lp_sts ./ 1e9, cs_nlp_sts ./ 1e9]);
title("Mean solve time")
xlabel('Time horizon (time steps)')
ylabel('Solve time (sec)')
legend('Linear MPC', 'Nonlinear MPC')


function err = error_norm(mat)
    abs_error = sqrt(mat(:, 30).^2 + mat(:, 31).^2 + mat(:, 32).^2);
    err = mean(abs_error);
end