function kb = p2_modelSizeKB(model)
%P2_MODELSIZEKB  Size of a trained model when saved to disk (kB).

f = [tempname '.mat'];
mdl = model.mdl; %#ok<NASGU>
save(f, "mdl");
d = dir(f);
kb = d.bytes/1024;
delete(f);
end
