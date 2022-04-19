function y = Infill_EI(x,model,fmin)

[u,s] = kriging_predictor(x,model);
y = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
end




