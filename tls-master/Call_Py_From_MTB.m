
%pyversion C:\ProgramData\Anaconda3\envs\python356env\python.exe ;
%py.importlib.importmodule('C:\Users\Arian\sample\ThesisCode\Python\py_integral_gpu')
%rmpath('C:\Users\Arian\sample\ThesisCode\Python\')
py_lib = 'C:\Users\Arian\sample\ThesisCode\Python\';
count(py.sys.path,py_lib)
if count(py.sys.path,py_lib) == 0
    insert(py.sys.path,int32(0),py_lib);
    py.importlib.import_module('Mylib');
end
%insert(py.sys.path,int32(0),py_lib);
%py.importlib.import_module('Mylib');
%insert(py.sys.path,int32(0),'C:\Users\Arian\sample\ThesisCode\Python\')
%py.sys.path.remove('C:\Users\Arian\sample\ThesisCode\Python\')
%py.sys.path

%py.importlib.import_module('py_integral_gpu.py')
%pyversion
%py.print('salamss')
%py.Mylib.foo('C:\Users\Arian\sample\ThesisCode\SharedAreaMtbPy\F.mat')
py.Mylib.fuu(reshape(randn(100,100).',1,[]))
%py.Mylib.fuu_test()
%py.Mylib.fuu_test()
%py.Mylib.fuu_test()
%py.Mylib.fuu_test()
%py.py_integral_gpu.foo(py.str('C:\Users\Arian\sample\ThesisCode\SharedAreaMtbPy\F.mat'))
% if count(py.sys.path,'') == 0
%     insert(py.sys.path,int32(0),'');
% end
%  py.py_integral_gpu.py_integral_gpu('C:\Users\Arian\sample\ThesisCode\SharedAreaMtbPy')
