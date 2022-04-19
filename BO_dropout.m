clearvars;clc;close all;

for name = {'Ellipsoid','Rosenbrock','Griewank','Ackley','Rastrigin'}
    fun_name = char(name);
    num_vari = 100;
    sub_vari = 40; 
    num_initial = 200;
    max_evaluation = 500;
    for i = 1 : 10        
        tstart=tic;
        if strcmp(fun_name, 'Ellipsoid')
            lower_bound = -5.12*ones(1,num_vari);
            upper_bound = 5.12*ones(1,num_vari);
        elseif strcmp(fun_name, 'Rosenbrock')
            lower_bound = -2.048*ones(1,num_vari);
            upper_bound = 2.048*ones(1,num_vari);
        elseif strcmp(fun_name, 'Griewank')
            lower_bound = -600*ones(1,num_vari);
            upper_bound = 600*ones(1,num_vari);
        elseif strcmp(fun_name, 'Ackley')     
            lower_bound = -32.768*ones(1,num_vari);
            upper_bound = 32.768*ones(1,num_vari);
        elseif strcmp(fun_name, 'Rastrigin')
            lower_bound = -5.12*ones(1,num_vari);
            upper_bound = 5.12*ones(1,num_vari);
        end
        sample_x = lhsdesign(num_initial, num_vari,'criterion','maximin','iterations',1000).*(upper_bound - lower_bound) + lower_bound;
        sample_y = feval(fun_name,sample_x);
        iteration = 1;
        evaluation =  size(sample_x,1);
        [fmin,ind] = min(sample_y);
        best_x = sample_x(ind,:);
        fmin_record(iteration,i) = fmin;
        fprintf('BO-dropout on %d-D %s,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
        while evaluation < max_evaluation
            rand_index = randperm(num_vari,sub_vari);
            sample_x_sub = sample_x(:,rand_index);
            tstart1=tic;
            kriging_model = kriging_train(sample_x_sub,sample_y,lower_bound(rand_index),upper_bound(rand_index),1*ones(1,sub_vari),0.01*ones(1,sub_vari),100*ones(1,sub_vari));
            [optimal_x,max_EI]= Optimizer_GA(@(x)-Infill_EI(x,kriging_model,fmin),sub_vari,lower_bound(rand_index),upper_bound(rand_index),5*sub_vari,200);
            infill_x = best_x;
            infill_x(rand_index)= optimal_x;
            infill_y = feval(fun_name, infill_x);
            iteration = iteration + 1;
            sample_x = [sample_x;infill_x];
            sample_y = [sample_y;infill_y];
            [fmin,ind]= min(sample_y);
            best_x = sample_x(ind,:);
            fmin_record(iteration,i) = fmin;
            evaluation = evaluation + size(infill_x,1);
            fprintf('BO-dropout on %d-D %s,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
            tend = toc(tstart);
        end
    end
    save(strcat('BO_40_',fun_name,'.mat'),'fmin_record');
    save(strcat('BO_40_total_time_',fun_name,'.mat'),'tend');
    save(strcat('BO_40_model_time_',fun_name,'.mat'),'count_time');
end
    
