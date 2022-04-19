clearvars;clc;close all;

for name = {'Ellipsoid','Rosenbrock','Griewank','Ackley','Rastrigin'}
    fun_name = char(name);
    num_vari = 100;
    num_initial = 200;
    max_evaluation = 500;
    fmin_record = zeros(max_evaluation-num_initial+1,10);
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
        [fmin,ind]= min(sample_y);
        best_x = sample_x(ind,:);
        fmin_record(iteration) = fmin;
        fprintf('BO on %d-D %s,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
        while evaluation < max_evaluation
            tstart1=tic;
            kriging_model = kriging_train(sample_x,sample_y,lower_bound,upper_bound,ones(1,num_vari),0.01*ones(1,num_vari),100*ones(1,num_vari));
            tend1 = toc(tstart1);
            count_time(iteration) = tend1; 
            [optimal_x,max_EI]= Optimizer_GA(@(x)-Infill_EI(x,kriging_model,fmin),num_vari,lower_bound,upper_bound,5*num_vari,200);
            infill_x  = optimal_x;
            infill_y = feval(fun_name, infill_x);
            iteration = iteration + 1;
            sample_x = [sample_x;infill_x];
            sample_y = [sample_y;infill_y];
            [fmin,ind]= min(sample_y);
            best_x = sample_x(ind,:);
            fmin_record(iteration) = fmin;
            evaluation = evaluation + size(infill_x,1);
            fprintf('BO on %d-D %s,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
        end
        tend = toc(tstart);
        save(strcat('BO_',fun_name,'.mat'),'fmin_record');
        save(strcat('BO_time_',fun_name,'.mat'),'tend');
        save(strcat('BO_model_time1_',fun_name,'.mat'),'count_time');
        
end

