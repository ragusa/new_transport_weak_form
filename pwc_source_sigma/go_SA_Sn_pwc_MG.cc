//code to compute d\phi = \phi_{Sn} - \phi_{SPn}

# define Dimension 3
# define RESIDUAL false

 #include <legendre.h>

 #include <grid/tria.h>
 #include <grid/tria_boundary_lib.h>
 #include <grid/grid_out.h>
 #include <grid/intergrid_map.h>
 #include <grid/grid_generator.h>
 #include <grid/tria_accessor.h>
 #include <grid/tria_iterator.h>
 #include <grid/grid_refinement.h>
 #include <grid/grid_tools.h>

   
 #include <fe/fe_q.h>
 #include <fe/fe_values.h>
 #include <fe/mapping_q.h>
 #include <fe/fe_system.h>
 #include <fe/fe_dgq.h>
 
 #include <dofs/dof_tools.h>
 #include <dofs/dof_handler.h>
 #include <dofs/dof_accessor.h>
 #include <dofs/dof_constraints.h>
 #include <dofs/dof_renumbering.h>
 
 #include <base/function.h>
 #include <base/quadrature_lib.h>
 #include <base/logstream.h>
 #include <base/smartpointer.h>
 #include <base/convergence_table.h>
 #include <base/timer.h>
 #include <base/thread_management.h>
 #include <base/parameter_handler.h>
 #include <base/utilities.h>
 
 #include <numerics/data_out.h>
 #include <numerics/vectors.h>
 #include <numerics/matrices.h>
 #include <numerics/error_estimator.h>
 #include <numerics/solution_transfer.h>
 #include <numerics/fe_field_function.h>
 
 #include <lac/vector.h>
 #include <lac/full_matrix.h>
 #include <lac/sparse_matrix.h>
 #include <lac/solver_cg.h>
 #include <lac/precondition.h>
 #include <lac/identity_matrix.h>
 #include <lac/sparsity_pattern.h>
 #include <lac/compressed_sparsity_pattern.h>
 
 
 
 #include <fstream>
 #include <iostream>
 #include <cmath>
 #include <typeinfo>
 #include <sstream>
 #include <vector>
 #include <string>

 #define _USE_MATH_DEFINES
   
 using namespace dealii;  
 

 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class MaterialData
{
 public : 
  MaterialData(const std::string& filename);

   double get_total_XS(const unsigned int material_id, const unsigned i_group) const;
  double get_moment_XS(const unsigned int material_id, const unsigned i_group, const unsigned int i_moment) const;
  unsigned int get_n_materials () const;
  unsigned int get_n_moments () const;
  unsigned int get_n_groups () const;
 
 private :
  unsigned int n_materials;
  unsigned int n_moments;
  unsigned int n_groups;
  std::vector<std::vector<double> > total_XS;  
  std::vector<Table <2, double> > moment_XS;   
}; 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MaterialData::MaterialData(const std::string& filename)
{	// this function reads the material data
	std::ifstream f_in (filename.c_str());
	AssertThrow (f_in, ExcMessage ("Open material file failed!"));

	f_in >> n_materials;		// we read the number of media
	f_in >> n_groups;		// we read the number of groups
	f_in >> n_moments;		// we read the number of moments
		
	f_in.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
	
  total_XS.resize(n_groups);
  moment_XS.resize(n_groups);
  for(unsigned int g=0; g<n_groups; g++)
  {
    total_XS[g].resize(n_materials);
    moment_XS[g].reinit(n_materials, n_moments);
  }

	 
	for (unsigned int i=0; i<n_materials; i++)
	{
		double scatt_XS; //temp variable
		for (unsigned int g=0; g < n_groups; g++)
    {
		  f_in >> total_XS[g][i]; 			//we read the value of the total cross-sections
			for (unsigned int j=0; j< n_moments; j++)
			{
				f_in >> scatt_XS;	 	//we read the scattering cross-sections of each moment	
				moment_XS[g][i][j]=scatt_XS;
			}
		  f_in.ignore (std::numeric_limits<std::streamsize>::max(),'\n');
		}
	}
	
	f_in.close ();
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
double MaterialData::get_total_XS(const unsigned int material_id, const unsigned int group) const
{
 return total_XS[group][material_id]; 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
double MaterialData::get_moment_XS(const unsigned int material_id, const unsigned int group, const unsigned int i_moment) const
{
 return moment_XS[group][material_id][i_moment]; 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
unsigned int MaterialData::get_n_materials () const
{
	return n_materials;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
unsigned int MaterialData::get_n_moments () const
{
	return n_moments;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
unsigned int MaterialData::get_n_groups () const
{
 return n_groups;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
class BoundaryValues : public Function<dim>
{
 public :
  BoundaryValues() : Function<dim>() {};
  
  virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{ // we use this function when we have some Dirichlet boundary condition
 double value_boundary=0*p[0];
   
 return value_boundary; //  10 * p[0] = 10 * l'abscisse du point sur la bordure
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
class RHS : public Function<dim> // this class has the information that we need for the right-hand-side due to even parity components
{
 public :
  RHS() : Function<dim>() {};
  virtual double get_source (const Point<dim>   &p, unsigned int group, double sigma, std::vector<double> domain_size, const unsigned int  component = 0) const;
  virtual double get_Jinc (const Point<dim>   &p, double value, unsigned int group,
        unsigned int moment, std::vector<Tensor<1,dim> > Omega, std::vector<double> wt, const unsigned int  component = 0 ) const;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double RHS<dim>::get_source (const Point<dim> &p, unsigned int group, double sigma, std::vector<double> domain_size, const unsigned int /*component*/) const 
 { // we give the source for the direct problem
 double value = 0;

 
 //if(p[0] <= 2  && p[1] >=9 && p[1] <= 11 && p[2] >=9 && p[2] <= 11)  //3-D
 // value=1;          //3-D
 
 //if(p[0] <= 2  && p[1] >=9 && p[1] <= 11)  //2-D
 // value=1;

 double xlen = 20;
  
  double phi = 1.0;
  if((p[0]<=1.5&&p[0]>=0.5&&p[1]<=1.5&&p[1]>=0.5&&p[2]<=1.5&&p[2]>=0.5))  //debug, benchmark source
  value = phi/(4.0*M_PI);
 
 
 
 // if((p[0]<=11.0&&p[0]>=9.0&&p[1]<=11.0&&p[1]>=9.0))  //debug, benchmark source
 //  value =10.0/4.0/M_PI;

 return value; 
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double RHS<dim>::get_Jinc (const Point<dim> &p, double value, unsigned int group, unsigned int moment, std::vector<Tensor<1,dim> > Omega, std::vector<double> wt, const unsigned int /*component*/) const 
 { // we use that function when we have en incoming current to retrun the value
 	 Tensor<1,dim> n;
 	 n[0] = 1.0; n[1] = 0.0;
 	 if(dim == 3)
 	 	 n[2] = 0.0;
 	 double alpha = 0;    //renormalization factor
 	 double half_range_integral = value/(4*M_PI)*0.5*2.0*M_PI;        //actual half range current, auxiliary variable to compute \alpha
 	 double half_range_sum = 0;   //quadrature half range current, auxiliary variable to compute \alpha
 	 for(unsigned i=0; i<wt.size(); i++)
 	 {
 	 	 if(dim==2)
 	    half_range_sum += value/(4*M_PI)*sqrt(1-pow(Omega[i].norm(),2.0))*wt[i]*2; //Q-points for 2D only covers a quadrant
 	   // half_range_sum += value/(4*M_PI)*Omega[i][1]*wt[i]*2; //Q-points for 2D only covers a quadrant
 	   else if(dim==3)
 	   	 half_range_sum += value/(4*M_PI)*Omega[i][2]*wt[i];   //Q-point for 3D covers half sphere
 	 }
// 	 if(half_range_sum!=0.0)
// 	   alpha = half_range_integral/half_range_sum;
// 	 else
 	 	 alpha = 1.0;
 	 	 
// 	 cout<<"alpha = "<<alpha<<endl;   //debug

   double return_value = ( value/(4*M_PI) )*alpha;
   
   return_value = return_value;

   return return_value;  
   
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
class RHS_psi_minus : public Function<dim>  //this is the right_hand_side due to odd parity components
 {
 public :
  RHS_psi_minus(std::vector<Tensor<1,dim> > Omega, std::vector<double> wt);
   double get_source (const Point<dim>   &p, unsigned int group, std::vector<double> domain_size, const unsigned int  moment = 0) const;
 private :
  std::vector<Tensor<1,dim> > Omega;
  std::vector<double> wt;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
RHS_psi_minus<dim>::RHS_psi_minus(std::vector<Tensor<1,dim> > Omega, std::vector<double> wt)
 :
  Omega(Omega),
  wt(wt)
  {
  }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
double RHS_psi_minus<dim>::get_source (const Point<dim> &p, unsigned int group, std::vector<double> domain_size, const unsigned int moment) const 
 { // we give the source for the direct problem
        double value = 0;

        double phi = 1.0;
        double xlen = 20;
//        value = Omega[moment][0]*phi/(4*M_PI)*2.0*M_PI/xlen*cos(2.0*M_PI*(p[0]-xlen/2.0)/xlen);  //sin(x)sin(y) manufacturered solution
        
//        value = Omega[moment][0]*phi/(4*M_PI)*2.0*M_PI/xlen*cos(2.0*M_PI*(p[0]-xlen/2.0)/xlen)*sin(2.0*M_PI*(p[1]-xlen/2.0)/xlen)
//        			 +Omega[moment][1]*phi/(4*M_PI)*2.0*M_PI/xlen*sin(2.0*M_PI*(p[0]-xlen/2.0)/xlen)*cos(2.0*M_PI*(p[1]-xlen/2.0)/xlen);  //sin(x)sin(y) manufacturered solution
        
 
 //if(p[0] <= 2  && p[1] >=9 && p[1] <= 11 && p[2] >=9 && p[2] <= 11)  //3-D
 // value=1;          //3-D
 
// if(p[0] <= 2  && p[1] >=9 && p[1] <= 11)
  
 
 return value; 
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
class ResponseFunction : public Function<dim> // this is the class for the response function
{
 public :
  ResponseFunction() : Function<dim>() {};
  double get_value (const Point<dim>   &p, const unsigned int  component = 0) const;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double ResponseFunction<dim>::get_value(const Point<dim>   &p, const unsigned int  component) const
{
	double value = 0.0;
	for(unsigned int d=0; d<dim; d++)
	  if( p[d] >= 5 && p[d] <= 15)
		  value = 1.0;
		
	return value;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>            
class SN_group
{
 public :
  SN_group (const unsigned int          group,
     const MaterialData         &material_data,
     const Triangulation<dim>  &coarse_grid,
     const FiniteElement<dim>   &fe);
        ~SN_group();   
  
  void matrix_reinit();   // we reinitialize the matrices that we use
  void setup_system (const std::vector<SN_group<dim>*> sn_group, unsigned int m); // we create the pattern of the matrices that we use
  void assemble_scattering_matrix (const unsigned int n, const unsigned int m, double coefficient); // the different moments


  void solve (unsigned int n_iteration_cg, double convergence_tolerance, Vector<double> &solution_moment); // we solve the system of equation that we have
  void output (unsigned int m) const;         // we make the output for one moment for the direct and the adjoint problem
    
  
  Triangulation<dim>    triangulation; 
  const FiniteElement<dim>    &fe;
  DoFHandler<dim>     dof_handler;  
  ConstraintMatrix        hanging_node_constraints;
  std::vector<Vector<double> >     solution_moment; 
  Vector<double>     system_rhs;
  SparsityPattern     sparsity_pattern;
  SparseMatrix<double>   system_matrix;
  SparseMatrix<double>   system_psi_matrix; //used for reconstruction residual computation
  SparseMatrix<double>   system_scat_matrix; //used for reconstruction residual computation
  std::vector<SparseMatrix<double> >   scattering_matrix; //scattering matrix for various moments, reused for various directions
  MappingQ1<dim>     mapping;
  unsigned int       n_dofs;

  Vector<float> estimated_error_per_cell, adjoint_estimated_error_per_cell, new_estimated_error_per_cell;
 
 
 private :
  unsigned int			 group;
  unsigned int       n_moments;  //number of moments(directions) in current AMG level
	unsigned int       n_groups;
	unsigned int       n_Omega;
  const MaterialData              &material_data;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SN_group<dim>::SN_group(const unsigned int          group,  // this is the constructor of the SN_group object, we give some values by default
                            const MaterialData         &material_data,
                            const Triangulation<dim>   &coarse_grid,
                            const FiniteElement<dim>   &fe)
                 :
     fe (fe),
     dof_handler (triangulation),
                 group (group),
                 material_data (material_data)
    
 {
   triangulation.copy_triangulation (coarse_grid);
   n_groups = material_data.get_n_groups(),
	 n_moments = material_data.get_n_moments();
   dof_handler.distribute_dofs (fe);
   n_dofs = dof_handler.n_dofs(),
   n_Omega = (dim==2 ? 2*(1+n_moments/2)*n_moments/2
                     : 4*(1+n_moments/2)*n_moments/2);
   solution_moment.resize(n_Omega);
   for(unsigned int m=0; m<n_Omega; m++)
     solution_moment[m].reinit(n_dofs, false); 
   scattering_matrix.resize(n_moments);
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SN_group<dim>::~SN_group()
{// this is the destructor of the SN_group object
  dof_handler.clear(); 
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN_group<dim>::assemble_scattering_matrix(const unsigned int n, const unsigned int m, double coefficient)
            
{ 
  const QGauss<dim>  quadrature_formula (fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula, 
       update_values  |  update_JxW_values);
  const unsigned int n_q_points = fe_values.n_quadrature_points; 
  
  typename DoFHandler<dim>::active_cell_iterator	
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
  
  for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
    fe_values.reinit (cell);

    FullMatrix<double>   local_mass_matrix (fe.dofs_per_cell, fe.dofs_per_cell);
    local_mass_matrix=0;

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
     for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
        local_mass_matrix(i,j) += ( //we take the cross section of the moment n because the right-hand-side needs this moment
              coefficient*material_data.get_moment_XS(cell->material_id(), group, n)*
              fe_values.shape_value(i,q_point) *
              fe_values.shape_value(j,q_point) *
              fe_values.JxW(q_point));
     

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    cell->get_dof_indices (local_dof_indices);
  

    for (unsigned int i=0; i<dofs_per_cell; ++i)   // we add the value of the celle in the global matrix. We still must multiply this matrix by the solution
     for (unsigned int j=0; j<dofs_per_cell; ++j)  // for the associated moment. We do that in assemble_system and assemble_adjoint_system
      scattering_matrix[n].add (local_dof_indices[i], // pour avoir le membre de droite, il faut encore multiplier la matrice globale par 
          local_dof_indices[j],    
          local_mass_matrix(i,j));     
  }
    
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN_group<dim>::setup_system (const std::vector<SN_group<dim>*> sn_group, unsigned int m)
{ // here we give the pattern of all the matrices and the right-hand-side that we use 
 dof_handler.distribute_dofs (fe);
  
 sparsity_pattern.reinit (dof_handler.n_dofs(),
                           dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern); // we create the pattern of the matrix
     
    solution_moment[m].reinit (n_dofs, false); //we create the pattern of the solution
 
    system_rhs.reinit (dof_handler.n_dofs()); //we create the pattern of the right-hand-side
 
 hanging_node_constraints.clear ();      // here we take care about the hanging nodes
 DoFTools::make_hanging_node_constraints (dof_handler,
                                            hanging_node_constraints);

 hanging_node_constraints.close ();
 hanging_node_constraints.condense (sparsity_pattern);
           
  sparsity_pattern.compress(); 
 
  system_matrix.reinit (sparsity_pattern); // now the matrix receive her pattern
  system_psi_matrix.reinit (sparsity_pattern); // now the matrix receive her pattern
  system_scat_matrix.reinit (sparsity_pattern); // now the matrix receive her pattern

 scattering_matrix.resize (n_moments);
 for (unsigned int n=0; n<n_moments; n++)
  { 
   scattering_matrix[n].reinit (sparsity_pattern); 
  }
 }
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN_group<dim>::matrix_reinit()
{ // we just reinitialize the all the matrix and the right-hand-side that we use in the code
 system_rhs.reinit (dof_handler.n_dofs());
 system_matrix.reinit (sparsity_pattern);
 system_psi_matrix.reinit (sparsity_pattern);
 system_scat_matrix.reinit (sparsity_pattern);
 
 for (unsigned int n=0; n<n_moments; n++)
   scattering_matrix[n].reinit (sparsity_pattern); 
} 
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
template <int dim>
void SN_group<dim>::solve (unsigned int n_iteration_cg, double convergence_tolerance, Vector<double> &solution_moment)
{ // here we just solve the system of equation thanks to CG we use a SSOR preconditioner 
 SolverControl           solver_control (n_iteration_cg, convergence_tolerance);
 SolverCG<>              cg (solver_control);
    
 PreconditionSSOR<> preconditioner; 
 preconditioner.initialize(system_matrix, 1.2);  
 cg.solve (system_matrix, solution_moment, system_rhs,
            preconditioner); 

 hanging_node_constraints.distribute(solution_moment); 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>  
void SN_group<dim>::output (unsigned int m) const
 { //make the output for each moment  for the direct and the adjoint problem
 DataOut<dim> data_out;
 
 data_out.attach_dof_handler (dof_handler);
 data_out.add_data_vector (solution_moment[m], "solution");
 
 data_out.build_patches ();
 
 std::ostringstream filename;
 
 filename << "solution_direction-" << m << "-group-" << group << ".vtk";
 
 std::ofstream output (filename.str().c_str());
 data_out.write_vtk (output);
 }

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>            
class DSA
{
 public :
  DSA ();
  DSA ( const MaterialData         &material_data,
     const std::vector<int>   boundary_conditions,
     const std::vector<double>  boundary_value,
     const Triangulation<dim>  &coarse_grid,
     const FiniteElement<dim>   &fe,
          const std::vector<Tensor<1, dim> >  &Omega,
     const std::vector<double>  &wt);
   ~DSA();   
  
  void matrix_reinit();   // we reinitialize the matrices that we use, PROBABLY NOT NEEDED
  void setup_system (); // we create the pattern of the matrices that we use
  void assemble_system(Vector<double> &phi_old, Vector<double> &phi, unsigned int group);
  void run(Vector<double> &phi_old, Vector<double> &phi, unsigned int n_iteration_cg, double convergence_tolerance, unsigned int group);

  void solve (unsigned int n_iteration_cg, double convergence_tolerance); // we solve the system of equation that we have
  
  Triangulation<dim>    triangulation; 
  const FiniteElement<dim>    &fe;
  DoFHandler<dim>     dof_handler;  
  ConstraintMatrix       hanging_node_constraints;
  Vector<double>     solution_dphi;   //storage for correction for scalar flux after DSA calculation. It is the solution to DSA solve.
  Vector<double>     system_rhs;
  SparsityPattern     sparsity_pattern;
  SparseMatrix<double>   system_matrix; 
  SparseMatrix<double>   system_rhs_matrix; 
  MappingQ1<dim>     mapping;

 
 private :
  const MaterialData              &material_data;
  std::vector<int>      boundary_conditions;
  std::vector<double>     boundary_value;
  const std::vector<Tensor<1, dim> >     &Omega;
  const std::vector<double>     &wt;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
DSA<dim>::DSA(// this is the constructor of the DSA object, we give some values by default
                            const MaterialData         &material_data,
                            const std::vector<int>   boundary_conditions,
              const std::vector<double>  boundary_value,
                            const Triangulation<dim>   &coarse_grid,
                            const FiniteElement<dim>   &fe,
                            const std::vector<Tensor<1, dim> >  &Omega,
              const std::vector<double>  &wt)
                 :
     fe (fe),
     dof_handler (triangulation),
     material_data (material_data),
     boundary_conditions(boundary_conditions),
     boundary_value(boundary_value),     
                 Omega(Omega),
                 wt(wt)    
 {
   triangulation.copy_triangulation (coarse_grid);
   dof_handler.distribute_dofs (fe);
   solution_dphi.reinit(dof_handler.n_dofs(),false);
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
DSA<dim>::~DSA()
{// this is the destructor of the SN_group object
  dof_handler.clear(); 
  cout<<"DSA desctruted"<<endl;  //debug
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void DSA<dim>::matrix_reinit()
{ // we just reinitialize the all the matrix and the right-hand-side that we use in the code
 system_rhs.reinit (dof_handler.n_dofs());
 system_matrix.reinit (sparsity_pattern);
 system_rhs_matrix.reinit (sparsity_pattern);
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void DSA<dim>::setup_system ()
{ // here we give the pattern of all the matrices and the right-hand-side that we use 
 dof_handler.distribute_dofs (fe);
  
 sparsity_pattern.reinit (dof_handler.n_dofs(),
                           dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern); // we create the pattern of the matrix
     
    solution_dphi.reinit (dof_handler.n_dofs()); //we create the pattern of the solution
 
    system_rhs.reinit (dof_handler.n_dofs()); //we create the pattern of the right-hand-side
 
 hanging_node_constraints.clear ();      // here we take care about the hanging nodes
 DoFTools::make_hanging_node_constraints (dof_handler,
                                            hanging_node_constraints);

 hanging_node_constraints.close ();
 hanging_node_constraints.condense (sparsity_pattern);
           
    sparsity_pattern.compress(); 
 
    system_matrix.reinit (sparsity_pattern); // now the lhs matrix receive her pattern
    system_rhs_matrix.reinit (sparsity_pattern); // now the rhs matrix receive her pattern
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void DSA<dim>::assemble_system (Vector<double> &phi_old, Vector<double> &phi, unsigned int group)
{ // this function is used to make the the system of equations for the direct problem
 
 QGauss<dim>  quadrature_formula(fe.degree+1); 
  FEValues<dim> fe_values (fe, quadrature_formula, 
        update_values | update_gradients | update_q_points | update_JxW_values); 
 
 const QGauss<dim-1> face_quadrature_formula(fe.degree +1);  
 const unsigned int n_face_q_points = face_quadrature_formula.size();
 FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
        update_values | update_q_points | update_normal_vectors | update_JxW_values);

 const unsigned int   dofs_per_cell = fe.dofs_per_cell; 
 const unsigned int   n_q_points    = quadrature_formula.size(); 
 
 FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell); 
 FullMatrix<double>   cell_rhs_matrix (dofs_per_cell, dofs_per_cell);
 Vector<double>       cell_rhs (dofs_per_cell); 
 std::vector<unsigned int> local_dof_indices (dofs_per_cell);

 std::vector<int>::iterator boundary_exist;     //we check if we have some Robin boundary condtions
 boundary_exist=std::find(boundary_conditions.begin(), boundary_conditions.end(), 2);
 bool RC_exist=false;
 
 Vector<double> dphi = phi;
 
 double mu_bar = 0.0;  // <mu>,  used in the boundary condition
 mu_bar = 0.5;  //assume the integration over half range is accurate
 
 
 typename DoFHandler<dim>::active_cell_iterator 
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  
 for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
    {
  fe_values.reinit (cell);
 
  cell_matrix = 0;
  cell_rhs_matrix = 0;
  cell_rhs = 0;
  
  double st = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
  double diffusion_coefficient = 1.0/(3.0*st);   // we create the diffusion coefficient
  double ss = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0-moment of the scattering cross-section
  double sa = st - ss;  //calculate the absorption cross-section
  
  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
   for (unsigned int i=0; i<dofs_per_cell; ++i)
   {
    for (unsigned int j=0; j<dofs_per_cell; ++j)  //we give the values to the matrix
    {
     cell_matrix(i,j) += ((diffusion_coefficient *
         fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) +
         sa *         
         fe_values.shape_value (i,q_point) * fe_values.shape_value (j,q_point)) *
         fe_values.JxW (q_point));

     // assemble the matrix for the scattering residual on the RHS
      cell_rhs_matrix(i,j) += (ss*
          fe_values.shape_value (i, q_point) * fe_values.shape_value (j, q_point) * 
          fe_values.JxW (q_point) );
    }
   }    
   

  //we put the Robin boundary conditions if they exist
  if (boundary_exist != boundary_conditions.end()) 
  {
   RC_exist=true;
   for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)   // we make a loop over all the face, if we have a face where we have a boundary 
                       // condition we go to the if
    if (cell->at_boundary(face) && (boundary_conditions[cell->face(face)->boundary_indicator()] == 2))
    {
     unsigned int side = cell->face(face)->boundary_indicator();
     fe_face_values.reinit (cell, face);
        
     for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) // we modify the matrix because of the boundary condition
     {                                 // right-hand-side will not be changed since in DSA we can only have zero source on the boundaries
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      { 
       for (unsigned int j=0; j<dofs_per_cell; j++)  
        cell_matrix(i,j) +=  (mu_bar *
             fe_face_values.shape_value(i,q_point) *
             fe_face_values.shape_value(j,q_point)*
             fe_face_values.JxW(q_point));
      }
     }
    }
  }
 
  cell->get_dof_indices (local_dof_indices);

  for (unsigned int i=0; i<dofs_per_cell; ++i) // we put the matrix and the right-hand-side of the cell into the matirx and the right-hand-side of the system
  { 
   for (unsigned int j=0; j<dofs_per_cell; ++j)
   {
    system_matrix.add (local_dof_indices[i],
         local_dof_indices[j],
         cell_matrix(i,j));
         
    system_rhs_matrix.add (local_dof_indices[i],
         local_dof_indices[j],
         cell_rhs_matrix(i,j));
   }
  } 
  }
  system_rhs_matrix.vmult_add(system_rhs, dphi -= phi_old);  //build the RHS scattering residual source


 // we take care about the hanging nodes
 hanging_node_constraints.condense (system_matrix);  
 hanging_node_constraints.condense (system_rhs);

}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
template <int dim>
void DSA<dim>::solve (unsigned int n_iteration_cg, double convergence_tolerance)
{ // here we just solve the system of equation thanks to CG we use a SSOR preconditioner 
 SolverControl           solver_control (n_iteration_cg, convergence_tolerance);
 SolverCG<>              cg (solver_control);
    
 PreconditionSSOR<> preconditioner; 
 preconditioner.initialize(system_matrix, 1.2);  
 cg.solve (system_matrix, solution_dphi, system_rhs,
            preconditioner); 

 hanging_node_constraints.distribute(solution_dphi); 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void DSA<dim>::run(Vector<double> &phi_old, Vector<double> &phi,
         unsigned int n_iteration_cg, double convergence_tolerance, unsigned int group )
{
 std::cout<<"We begin to solve the DSA equation: "<<std::endl;
 setup_system (); // we create the matrix and the right-hand-side 
 assemble_system(phi_old, phi, group); // we give the values of the matrix and the right-hand-side

 solve(n_iteration_cg, convergence_tolerance); // we solve the system that we created just before   
 
 phi += solution_dphi;  //add the DSA correction to the scalar flux
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
class SN // here we define the SN class which contains the Parameters class
{
 public :
 
  class Parameters 
   {
   public:
    Parameters ();
 
    static void declare_parameters (ParameterHandler &prm);
    void get_parameters (ParameterHandler &prm);

    unsigned int pdim;       // dimension of the problem
    bool go;         // true if we have to make a goal oriented calculation
    bool all_output;
    bool current;
    unsigned int n_points;
    std::vector<double> point; 
    unsigned int n_refinement_cycles;     //number of refinement cycles
    unsigned int refinement_level_difference_limit; // limit for refinement levels on different meshes
    unsigned int n_iteration_cg;       //maximum number of iterations for CG
    double convergence_tolerance;   //criterion of convergence fo CG
    double conv_tol;    // critere de convergence sur les moments
    unsigned int fe_degree;     // degree of the finite elements
  
    std::string geometry_input_file;
    std::string assembly_file;
    std::string material_data_file;
    std::string output_file;
    std::string spn_output_file;

    std::vector<unsigned int> n_assemblies;  //number of assemblies
    std::vector<int> boundary_conditions;  //vector to know which kind of boundary conditions we have for the direct problem
    std::vector<double> boundary_value;   // value of the bondary if we have some Robin conditions
    std::vector<int> adjoint_boundary_conditions;  //vector to know which kind of boundary conditions we have for the adjoint problem
    std::vector<double> adjoint_boundary_value;   // value of the bondary if we have some Robin conditions

    std::vector<int> core_arrangement;   

    bool with_assemblies;
    std::vector<double> x_lengths;
    std::vector<unsigned int> x_subdivisions;
    std::vector<double> y_lengths;
    std::vector<unsigned int> y_subdivisions;
    std::vector<double> z_assembly_heights;
    std::vector<unsigned int> z_assembly_subdivisions;
  
    double refine_fraction;
    double coarsen_fraction;
   };
  
  SN(Parameters &prm);
  ~SN();
  void run ();
  Parameters                    &parameters;
  
 
 private :
  
  void assemble_system (unsigned int group, unsigned int moment); // make the system of equations for the direct problem
  void output (int cycle) const;   // we make the output of the solution
  void compute_phi(unsigned int group);
  void compute_response(unsigned int group);         //response := leakage for the boundaries flagged "incoming" in adjoint boundary conditions    
  void check_conservation(unsigned int group);     
  
  const MaterialData          material_data;
  Table<2,std::vector<unsigned int> >   coremap;
  const Triangulation<dim>          coarse_grid;
  
  FE_Q<dim>      fe;
  std::vector<double>    domain_size;  //dimension of the problem along x,y,z

  Triangulation<dim>     initial_grid();
  
  void read_SPn_data ();                   //read-in SPn flux moment output data
  void reconstruct_SPn_psi_plus();              //Reconstruct even-parity Angular flux from SPn solution
  void compute_reconstruct_phi();            //Reconstruct scalar flux from reconstructed psi+
  Table<2, Vector<double> >      phi_even_spn;    //phi even moment from SPn code;
  Table<2, Vector<double> >      phi_odd_spn;     //phi even moment from SPn code;
  Table<2, Vector<double> >      psi_plus_recon;  //psi+ reconstrcuted from SPn solution
  std::vector<Vector<double> >   phi_spn_recon;  //Reconstructed scalar flux
  std::vector<Vector<double> >   dof_repitition;  //Shared DOF repitition counter
  std::vector<std::vector<std::vector<unsigned int> > > dof_neighbour_dof; //neighbouring dofs around a certian dof
  
  std::vector<Vector<double> >   solution;  //angluar flux
  std::vector<double>              response;  // <\phi, R>
  std::vector<Vector<double> >      phi;  //scalar flux
  std::vector<Vector<double> >     phi_old;  //scalar flux from previous SI iteration
  
  std::vector<Tensor<1,dim> >         Omega;  //Sn directions(angular quadrature points)
  std::vector<double>         wt;  //weight corresponding to Omega
  unsigned int    n_Omega;   //number of discrete directions 
  unsigned int    n_groups;

  std::ofstream               outp;
  Triangulation<dim>    common_triangulation;
  MappingQ1<dim>     common_mapping;
 
  std::vector<SN_group<dim>*>    sn_group;
  DSA<dim>*   dsa;
}; 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SN<dim>::Parameters::Parameters()  // this is the constructor for the parameters objects, we give some value by default 
    :
    pdim(2),
    go(false),
    n_points(0),
    n_refinement_cycles(10),
    refinement_level_difference_limit (100),
    n_iteration_cg(1000),
    convergence_tolerance(1e-12),
    fe_degree(2),
    assembly_file ("assembly"),
    material_data_file ("material_data"),
    output_file ("outp"),
    spn_output_file("SPn_angular_flux"),
    refine_fraction(0.25),
    coarsen_fraction(0.01)    
{}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN<dim>::Parameters::declare_parameters (ParameterHandler &prm)
{
  //this function declares all the entries in project.prm These entries we'll be used to give the good values to the parameters of our program
  prm.declare_entry ("Dimension", "2",
       Patterns::Integer (),
       "Dimension of the problem");
  prm.declare_entry ("Goal oriented", "false",
       Patterns::Bool (),
       "Use of the goal oriented");
  prm.declare_entry ("Current", "false",
       Patterns::Bool (),
       "True if the quantity of interest is the current");
  prm.declare_entry ("Number of interesting points", "0",
       Patterns::Integer (),
       "Number of points where we have an quantity of interest");
  prm.declare_entry ("Point", "",
       Patterns::List (Patterns::Double()),
       "We give the coordinate of the point where we want to know the quantity of interest");
  prm.declare_entry ("All output", "false",
       Patterns::Bool (),
       "We show all the ouptuts");
  prm.declare_entry ("Refinement cycles", "5",
       Patterns::Integer (),
       "Number of refinement cycles to be performed");
  prm.declare_entry ("Refinement level difference limit", "100",
       Patterns::Integer (),
       "Number of levels by which different meshes can differ");
  prm.declare_entry ("Finite element degree", "2",
       Patterns::Integer (),
       "Polynomial degree of the finite element to be used");
  prm.declare_entry ("Convergence tolerance", "1e-12",
       Patterns::Double (),
       "CG iterations are stopped when the change in the solution falls "
       "below this tolerance");
  prm.declare_entry ("Maximum number of CG iterations", "500",
       Patterns::Integer (),
       "CG iterations are stopped when the iteration number exceeds "
       "this number");
  prm.declare_entry ("Convergence tolerance for the moments", "1e-7",
       Patterns::Double (),
       "We stop to iterate over all the moments when the change in the solution falls "
       "below this tolerance");
  prm.declare_entry ("Assembly file", "assembly",
       Patterns::Anything (),
       "Name of the assembly file");
  prm.declare_entry ("Material data file", "material_data",
       Patterns::Anything (),
       "Name of the material data input file");
  prm.declare_entry ("Output file", "outp",
       Patterns::Anything (),
       "Name of the output file");
 prm.declare_entry ("SPn data output file", "SPn_angular_flux",
       Patterns::Anything (),
       "Name of the SPn flux data output file");

  prm.declare_entry ("Number of assemblies", (dim == 2? "2,2" : "2,2,1"),
       Patterns::List (Patterns::Integer(1), dim, dim),
       "A list of length equal to the spatial dimension, which "
       "contains the number of assemblies in each of the space "
       "dimensions");

  prm.declare_entry ("Have assembly file", "true",
       Patterns::Bool(),
       "Indicate if assembly structures inside of the core arrangement.");
  prm.declare_entry ("X grid", "",
       Patterns::List (Patterns::Double()),
       "Grids in x-direction and number of cells each grid to be subdivided into.");
  prm.declare_entry ("Y grid", "",
       Patterns::List (Patterns::Double()),
       "Grids in y-direction and number of cells each grid to be subdivided into.");

  prm.declare_entry ("Z assembly description", "",
       Patterns::List (Patterns::Double()),
       "A description of how assemblies are stacked in the z-direction. "
       "For each assembly in z-direction specified in the number of "
       "assemblies, there should be a pair of numbers first "
       "specifying the height of this assembly and then the number of "
       "cells this assembly is to be subdivided into. "
       "This field is ignored if we are not in 3-d.");
  
  prm.declare_entry ("Boundary conditions", (dim == 2? "2,2,2,2" : "2,2,2,2,2,2"),
       Patterns::List (Patterns::Integer(0), 2*dim, 2*dim),
       "List of boundary conditions for each of the 2*dim "
       "faces of the domain");
  prm.declare_entry ("Boundary values", "",
       Patterns::List (Patterns::Double()),
       "List of boundary values for each of the 2*dim "
       "faces of the domain");
  prm.declare_entry ("Adjoint boundary conditions", (dim == 2? "2,2,2,2" : "2,2,2,2,2,2"),
       Patterns::List (Patterns::Integer(0), 2*dim, 2*dim),
       "List of boundary conditions for each of the 2*dim "
       "faces of the domain");
  prm.declare_entry ("Adjoint boundary values", "",
       Patterns::List (Patterns::Double()),
       "List of boundary values for each of the 2*dim "
       "faces of the domain");
  prm.declare_entry ("Core arrangement", "",
       Patterns::List (Patterns::Integer(0)),
       "A list of core descriptors. The length of this list must "
       "be equal to the product of the number of assemblies in "
       "all the coordinate directions");
  prm.declare_entry ("Refinement fraction", "0.3",
       Patterns::Double (),
       "Refinement fraction of numbers of cells of each refinement cycle");
  prm.declare_entry ("Coarsening fraction", "0.01",
       Patterns::Double (),
       "Coarsening fraction of numbers of cells of each refinement cycle");
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN<dim>::Parameters::get_parameters (ParameterHandler &prm)
{ // here we read the project.prm and we give the good value for all the parameters
  pdim                    = prm.get_integer ("Dimension");
  AssertThrow (pdim == dim, 
        ExcMessage ("Problem dimension is not consistent with the code!"));
  go                          = prm.get_bool    ("Goal oriented");
  current        = prm.get_bool    ("Current");
  n_points       = prm.get_integer ("Number of interesting points");
  all_output                  = prm.get_bool    ("All output");
  n_refinement_cycles         = prm.get_integer ("Refinement cycles");
  refinement_level_difference_limit      = prm.get_integer ("Refinement level difference limit");
  fe_degree                   = prm.get_integer ("Finite element degree");
  convergence_tolerance    = prm.get_double  ("Convergence tolerance");
  conv_tol              = prm.get_double  ("Convergence tolerance for the moments");
  n_iteration_cg         = prm.get_integer ("Maximum number of CG iterations");
  assembly_file               = prm.get         ("Assembly file");
  material_data_file          = prm.get         ("Material data file");
  output_file                 = prm.get         ("Output file");
  spn_output_file             = prm.get         ("SPn data output file");


  
 std::vector<std::string> cp_tmp = (Utilities::split_string_list(prm.get ("Point")));
 AssertThrow (cp_tmp.size() == n_points*pdim,
 ExcMessage ("The number of coordinate for the point for interest is wrong"));
 
 for (unsigned int i=0; i<n_points*pdim; ++i)
  point.push_back ( std::atof( cp_tmp[i].c_str() ) );
 
  const std::vector<int>
    n_assemblies_int = Utilities::string_to_int
    (Utilities::split_string_list(prm.get ("Number of assemblies")));
  n_assemblies = std::vector<unsigned int>(n_assemblies_int.begin(),
        n_assemblies_int.end());
  boundary_conditions =
    Utilities::string_to_int
    (Utilities::split_string_list(prm.get ("Boundary conditions")));
 
 unsigned int n_robin =0;
 n_robin = count(boundary_conditions.begin(), boundary_conditions.end(), 2);
 
 std::vector<std::string> bv_tmp = (Utilities::split_string_list(prm.get ("Boundary values")));
 AssertThrow (bv_tmp.size() == n_robin,
 ExcMessage ("Description boundary value has wrong "
          "number of entries"));
 
 for (unsigned int i=0,j=0; i<boundary_conditions.size(); ++i)
 {
  if(boundary_conditions[i]==2)
  {
   boundary_value.push_back ( std::atof( bv_tmp[j].c_str() ) );
   j++;
  }
  else
  boundary_value.push_back (0);
 }
  
 adjoint_boundary_conditions =
    Utilities::string_to_int
    (Utilities::split_string_list(prm.get ("Adjoint boundary conditions")));
 
 n_robin = count(adjoint_boundary_conditions.begin(), adjoint_boundary_conditions.end(), 2);
 
 std::vector<std::string> abv_tmp = (Utilities::split_string_list(prm.get ("Adjoint boundary values")));
 AssertThrow (abv_tmp.size() == n_robin,
 ExcMessage ("Description adjoint boundary value has wrong "
          "number of entries"));
 
 for (unsigned int i=0,j=0; i<adjoint_boundary_conditions.size(); ++i)
 {
  if(adjoint_boundary_conditions[i]==2)
  {
   adjoint_boundary_value.push_back ( std::atof( abv_tmp[j].c_str() ) );
   j++;
  }
  else
  adjoint_boundary_value.push_back (0);
 }
 
  core_arrangement =
    Utilities::string_to_int
    (Utilities::split_string_list(prm.get ("Core arrangement")));

for(unsigned int i = 0; i<dim; i++)
   cout<<"n_assemblies["<<i<<"] = "<<n_assemblies[i]<<endl;  //debug
   cout<<"core arrangement size = "<<core_arrangement.size()<<endl;
   
  AssertThrow (core_arrangement.size() ==
        (dim == 2 ?
  n_assemblies[0] * n_assemblies[1] :
  n_assemblies[0] * n_assemblies[1] * n_assemblies[2]),
        ExcMessage ("Number of core descriptors incorrect"));

  with_assemblies = prm.get_bool ("Have assembly file");
  if (!with_assemblies)
  {
 std::vector<std::string> str_tmp = (Utilities::split_string_list(prm.get ("X grid")));
 AssertThrow (str_tmp.size() == n_assemblies[0] * 2,
 ExcMessage ("Description of x assemblies has wrong "
          "number of entries"));
 
 for (unsigned int i=0; i<n_assemblies[0]; ++i)
 {
  x_lengths.push_back ( std::atof( str_tmp[i*2].c_str() ) );
  x_subdivisions.push_back ( std::atoi( str_tmp[i*2+1].c_str() ) );
 }
  
 std::vector<std::string> stry_tmp = (Utilities::split_string_list(prm.get ("Y grid")));
 AssertThrow (stry_tmp.size() == n_assemblies[1] * 2,
 ExcMessage ("Description of y assemblies has wrong "
          "number of entries"));
 
 for (unsigned int i=0; i<n_assemblies[1]; ++i)
 {
  y_lengths.push_back ( std::atof( stry_tmp[i*2].c_str() ) );
  y_subdivisions.push_back ( std::atoi( stry_tmp[i*2+1].c_str() ) );
 }
 }

  if (dim == 3)
  {
 std::vector<std::string> str_tmp = (Utilities::split_string_list(prm.get ("Z assembly description")));
 AssertThrow (str_tmp.size() == n_assemblies[2] * 2,
 ExcMessage ("Description of z assemblies has wrong "
          "number of entries"));
 
 for (unsigned int i=0; i<n_assemblies[2]; ++i)
 {
  z_assembly_heights.push_back ( std::atof( str_tmp[i*2].c_str() ) );
  z_assembly_subdivisions.push_back ( std::atoi( str_tmp[i*2+1].c_str() ) );
 }  
  }
  
 refine_fraction = prm.get_double  ("Refinement fraction");
 coarsen_fraction = prm.get_double ("Coarsening fraction");
} 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SN<dim>::SN(Parameters &prm)  // constructor of the SN object 
   :
   parameters (prm),
            material_data (parameters.material_data_file),
   coarse_grid(initial_grid()),
            fe (parameters.fe_degree),       //scalar FE for psi+
   outp (parameters.output_file.c_str())
{ 
 unsigned int n_moments = material_data.get_n_moments();
 n_groups = material_data.get_n_groups();
	
 n_Omega = (dim==2 ? 2*(1+n_moments/2)*n_moments/2
                   : 4*(1+n_moments/2)*n_moments/2);
 sn_group.resize (n_groups);

  QGauss<1>  mu_quadrature(material_data.get_n_moments());  //get angular qudrature points according to the order of Sn approximation to be used.
 for (unsigned int moment=0; moment<n_moments; moment++)  
 {
 	int level;
 	if(moment < n_moments/2)
    level = moment + 1; //latitude level index above the equator
  else
  	level = n_moments/2 - (moment - n_moments/2); //latitude level index below the equator
  	
  cout<<"Level("<<level<<"):"<<endl;
  double mu = mu_quadrature.point(n_moments - (moment + 1))[0]*2.0-1.0;  //mu = cos(\theta), polar angle cosine
  Tensor<1, dim> temp;  //temporary Tensor to store angular qudrature point
  if(dim == 3)
    temp[2] = mu;		// \Omega_z
  double w0 = 2.0*M_PI/(4.0*level)/2.0;
  for(int n_w=0; n_w<(dim==2 ? 2*level : 4*level); n_w++)
  {
   double w = w0 + n_w*2.0*M_PI/(4.0*level);
   temp[0] = sqrt(1.0 - mu*mu)*cos(w);  //x component of the direction
   temp[1] = sqrt(1.0 - mu*mu)*sin(w);  //y compoment of the direction
   Omega.push_back(temp);
   wt.push_back(2.0*mu_quadrature.weight(n_moments - (moment + 1))*2.0*M_PI/(4.0*level));  //polar weights sum to 2, total weights sum to 4*pi 
   cout<<temp[0]<<" "<<temp[1]<<" "<<temp[2]<<endl;
   cout<<"wt = "<<2.0*mu_quadrature.weight(n_moments - (moment + 1))*2.0*M_PI/(4.0*level)<<endl;
  }
 }

 for (unsigned int group=0; group<n_groups; group++)
  sn_group[group] = new SN_group<dim> (group, material_data, coarse_grid, fe);

 dsa = new DSA<dim> (material_data, parameters.boundary_conditions, parameters.boundary_value, coarse_grid, fe, Omega, wt);  //initializing DSA object
 
  solution.resize(n_groups);	
	phi_even_spn.reinit(n_groups, n_moments);
	phi_odd_spn.reinit(n_groups, n_moments);
	psi_plus_recon.reinit(n_groups, n_Omega);
	phi_spn_recon.resize(n_groups);
	phi.resize(n_groups);
	phi_old.resize(n_groups);
	response.resize(n_groups);
  
  dof_neighbour_dof.resize(n_groups);
  dof_repitition.resize(n_groups);
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SN<dim>::~SN()
{

 //delete common_dof_handler;
 delete dsa;
 for (unsigned int group=0; group<n_groups; group++)
  delete sn_group[group];
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Triangulation<dim> SN<dim>::initial_grid()
// this function reads the file for the assembly if it exists and we create the initial grid of our problem thanks to the informations that we have. It's here that we give the material properties of the cells
{
  Assert (false, ExcMessage ("1-D assembly data not available!"));
}

#if Dimension == 2

 template <>
 Triangulation<2> SN<2>::initial_grid()
 {
  Triangulation<2> t;
  Table<2,unsigned int> core(parameters.n_assemblies[0], parameters.n_assemblies[1]);
  
  unsigned int index = 0;
  for (unsigned int j=parameters.n_assemblies[1]; j>0; j--)
   for (unsigned int i=0; i<parameters.n_assemblies[0]; i++)
    core[i][j-1] = parameters.core_arrangement[index++];
  if (parameters.with_assemblies)
  {
   std::ifstream a_in(parameters.assembly_file.c_str());
   AssertThrow (a_in, ExcMessage ("Open assembly file failed!"));
      
   unsigned int n_assembly_types;
   a_in >> n_assembly_types;
   a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
   unsigned int rods_per_assembly_x, rods_per_assembly_y;
   a_in >> rods_per_assembly_x>> rods_per_assembly_y;
   a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
   double pin_pitch_x, pin_pitch_y;
   a_in >> pin_pitch_x>> pin_pitch_y;
   a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
   Table<3,unsigned int> assembly_materials(n_assembly_types, rods_per_assembly_x, rods_per_assembly_y);
   for (unsigned int i=0; i<n_assembly_types; i++) 
   {
    a_in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    for (unsigned int j=rods_per_assembly_y; j>0; j--) 
    {
     for (unsigned int k=0; k<rods_per_assembly_x; k++)
      a_in >> assembly_materials[i][j-1][k];
     a_in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
   }
   a_in.close();
      
   const Point<2> bottom_left = Point<2>();
   const Point<2> upper_right = Point<2> (parameters.n_assemblies[0]*rods_per_assembly_x*pin_pitch_x,
          parameters.n_assemblies[1]*rods_per_assembly_y*pin_pitch_y);
      
   std::vector< std::vector<double> > n_subdivisions;
   std::vector<double> xv(parameters.n_assemblies[0]*rods_per_assembly_x,pin_pitch_x);
   n_subdivisions.push_back (xv);
   std::vector<double> yv(parameters.n_assemblies[1]*rods_per_assembly_y,pin_pitch_y);
   n_subdivisions.push_back (yv);  
   
   Table<2,unsigned char> material_id(rods_per_assembly_x*parameters.n_assemblies[0],
            rods_per_assembly_y*parameters.n_assemblies[1]);
   for (unsigned int j=0; j<parameters.n_assemblies[1];j++)
    for (unsigned int aj=0; aj<rods_per_assembly_y; aj++)
     for (unsigned int i=0; i<parameters.n_assemblies[0];i++)
      for (unsigned int ai=0; ai<rods_per_assembly_x; ai++)
       material_id[i*rods_per_assembly_x+ai][j*rods_per_assembly_y+aj] = (assembly_materials[core[i][j]-1][aj][ai]-1);

   GridGenerator::subdivided_hyper_rectangle (t, n_subdivisions, bottom_left, material_id, true);
   
   coremap.reinit (parameters.n_assemblies[0]*rods_per_assembly_x, parameters.n_assemblies[1]*rods_per_assembly_y);
   Triangulation<2>::raw_cell_iterator cell = t.begin_raw(), endc = t.end();
   for (unsigned int id=0; cell !=endc; ++cell,++id) 
   {
    Point<2> cell_center = cell->center();
    for (unsigned int j=0; j<parameters.n_assemblies[1]*rods_per_assembly_x; j++)
     for (unsigned int i=0; i<parameters.n_assemblies[0]*rods_per_assembly_y; i++) 
     {
      Point<2> p1(pin_pitch_x*i,pin_pitch_y*j);
      Point<2> p2(pin_pitch_x*(i+1),pin_pitch_y*(j+1));
      if (cell_center[0]>p1[0] && cell_center[0]<p2[0] && cell_center[1]>p1[1] && cell_center[1]<p2[1])
       coremap[i][j].push_back(id);
     }
   }
  }
  else
  {
   unsigned int nx = parameters.x_lengths.size ();
   unsigned int ny = parameters.y_lengths.size ();
   unsigned int mx = 0;
   unsigned int my = 0;

   std::vector< std::vector<double> > n_subdivisions;
   std::vector<double> xv;
   double xlen = 0;
   for (unsigned int i=0; i<nx; i++)
   {
    xlen += parameters.x_lengths[i];
    for (unsigned int j=0; j<parameters.x_subdivisions[i]; j++,mx++)
     xv.push_back (parameters.x_lengths[i]/parameters.x_subdivisions[i]);
   }
   n_subdivisions.push_back (xv);
   std::vector<double> yv;
   double ylen = 0;
   for (unsigned int i=0; i<ny; i++)
   {  
    ylen += parameters.y_lengths[i];
    for (unsigned int j=0; j<parameters.y_subdivisions[i]; j++,my++)
     yv.push_back (parameters.y_lengths[i]/parameters.y_subdivisions[i]);
   }
   n_subdivisions.push_back (yv);

   const Point<2> bottom_left = Point<2>();
   const Point<2> upper_right = Point<2> (xlen, ylen);

   Table<2,unsigned char> material_id(mx,my);
   for (unsigned int i=0, m=0; i<nx; i++)
    for (unsigned int ix=0; ix<parameters.x_subdivisions[i]; ix++,m++)
     for (unsigned int j=0, n=0; j<ny; j++)
      for (unsigned int iy=0; iy<parameters.y_subdivisions[j]; iy++,n++)
       material_id[m][n] = core[i][j]-1;

   GridGenerator::subdivided_hyper_rectangle (t, n_subdivisions, bottom_left, material_id, true);
       //t.refine_global(2);  //debug,  isotropically refine mesh

   coremap.reinit (nx,ny);
   Triangulation<2>::raw_cell_iterator cell = t.begin_raw(),
   endc = t.end();
   for (unsigned int id=0; cell !=endc; ++cell,++id)
   {
    Point<2> cell_center = cell->center();
    double xp = 0;
    unsigned int i=0;
    for (; i<nx; i++)
    {
     if (cell_center[0]>xp && cell_center[0]<xp+parameters.x_lengths[i]) break;
      xp += parameters.x_lengths[i];
    }
    xp = 0;
    unsigned int j=0;
    for (; j<ny; j++)
    {
     if (cell_center[1]>xp && cell_center[1]<xp+parameters.y_lengths[j]) break;
      xp += parameters.y_lengths[j];
    }

   coremap[i][j].push_back(id);
   }
  }
  return t;
 } 
#endif 

#if Dimension == 3

template <>
Triangulation<3> SN<3>::initial_grid()
{
  Triangulation<3> t;

  double t_height=0;
  unsigned int z_meshes = 0;
  
  for (unsigned int i=0; i<parameters.n_assemblies[2]; i++)
    {
      t_height += parameters.z_assembly_heights[i];
      z_meshes += parameters.z_assembly_subdivisions[i];
    }
  
      // unroll the list of read in
      // core descriptors
  Table<3,unsigned int> core(parameters.n_assemblies[0],
        parameters.n_assemblies[1],
        parameters.n_assemblies[2]);
  unsigned int index = 0;
  for (unsigned int k=0; k<parameters.n_assemblies[2]; k++)
    for (unsigned int j=parameters.n_assemblies[1]; j>0; j--)
      for (unsigned int i=0; i<parameters.n_assemblies[0]; i++)
 core[i][j-1][k] = parameters.core_arrangement[index++];

  if (parameters.with_assemblies)
    {
      std::ifstream a_in(parameters.assembly_file.c_str());
      AssertThrow (a_in, ExcMessage ("Open assembly file failed!"));
      
      unsigned int n_assembly_types;
      a_in >> n_assembly_types;
      a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
      unsigned int rods_per_assembly_x, rods_per_assembly_y;
      a_in >> rods_per_assembly_x
    >> rods_per_assembly_y;
      a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
      double pin_pitch_x, pin_pitch_y;
      a_in >> pin_pitch_x
    >> pin_pitch_y;
      a_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      
      Table<3,unsigned int> assembly_materials(n_assembly_types, rods_per_assembly_x, rods_per_assembly_y);
      for (unsigned int i=0; i<n_assembly_types; i++) {
 a_in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
 for (unsigned int j=rods_per_assembly_y; j>0; j--) {
   for (unsigned int k=0; k<rods_per_assembly_x; k++)
     a_in >> assembly_materials[i][j-1][k];
   a_in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
 }
      }
      a_in.close();
      
      const Point<3> bottom_left = Point<3>();
      const Point<3> upper_right = Point<3> (parameters.n_assemblies[0]*rods_per_assembly_x*pin_pitch_x,
          parameters.n_assemblies[1]*rods_per_assembly_y*pin_pitch_y,
          t_height);
  
      std::vector< std::vector<double> > n_subdivisions;
      std::vector<double> xv(parameters.n_assemblies[0]*rods_per_assembly_x,pin_pitch_x);
      n_subdivisions.push_back (xv);
      std::vector<double> yv(parameters.n_assemblies[1]*rods_per_assembly_y,pin_pitch_y);
      n_subdivisions.push_back (yv);
      std::vector<double> zv;
      unsigned int nz = 0;
      for (unsigned int i=0; i<parameters.n_assemblies[2]; i++)
 for (unsigned int j=0; j<parameters.z_assembly_subdivisions[i]; j++,nz++)
   zv.push_back(parameters.z_assembly_heights[i]/parameters.z_assembly_subdivisions[i]);
      n_subdivisions.push_back (zv);
  
      Table<3,unsigned char> material_id(rods_per_assembly_x*parameters.n_assemblies[0],
      rods_per_assembly_y*parameters.n_assemblies[1],
      nz);
      unsigned int zb = 0;
      for (unsigned int k=0; k<parameters.n_assemblies[2]; k++) {
 for (unsigned int ak=0; ak<parameters.z_assembly_subdivisions[k]; ak++)
   for (unsigned int j=0; j<parameters.n_assemblies[1];j++)
     for (unsigned int aj=0; aj<rods_per_assembly_y; aj++)
       for (unsigned int i=0; i<parameters.n_assemblies[0];i++)
  for (unsigned int ai=0; ai<rods_per_assembly_x; ai++)
    material_id[i*rods_per_assembly_x+ai]
      [j*rods_per_assembly_y+aj]
      [zb+ak] = (assembly_materials[core[i][j][k]-1][aj][ai]-1);
 zb += parameters.z_assembly_subdivisions[k];
      }

      GridGenerator::subdivided_hyper_rectangle (t,
       n_subdivisions,
       bottom_left,
       material_id,
       true);
      
      coremap.reinit (parameters.n_assemblies[0]*rods_per_assembly_x,
        parameters.n_assemblies[1]*rods_per_assembly_y);
      Triangulation<3>::raw_cell_iterator cell = t.begin_raw(),
 endc = t.end();
      for (unsigned int id=0; cell !=endc; ++cell,++id) {
 Point<3> cell_center = cell->center();
 

 for (unsigned int j=0; j<parameters.n_assemblies[1]*rods_per_assembly_x; j++)
   for (unsigned int i=0; i<parameters.n_assemblies[0]*rods_per_assembly_y; i++) {
     Point<3> p1(pin_pitch_x*i,pin_pitch_y*j,0);
     Point<3> p2(pin_pitch_x*(i+1),pin_pitch_y*(j+1),t_height);
     if (cell_center[0]>p1[0] && cell_center[0]<p2[0] &&
  cell_center[1]>p1[1] && cell_center[1]<p2[1]) coremap[i][j].push_back(id);
   }
      }
    }
  else
    {
      unsigned int nx = parameters.x_lengths.size ();
      unsigned int ny = parameters.y_lengths.size ();
      unsigned int mx = 0;
      unsigned int my = 0;

      std::vector< std::vector<double> > n_subdivisions;
      std::vector<double> xv;
      double xlen = 0;
      for (unsigned int i=0; i<nx; i++)
 {
   xlen += parameters.x_lengths[i];
   for (unsigned int j=0; j<parameters.x_subdivisions[i]; j++,mx++)
     xv.push_back (parameters.x_lengths[i]/parameters.x_subdivisions[i]);
 }
      n_subdivisions.push_back (xv);
      std::vector<double> yv;
      double ylen = 0;
      for (unsigned int i=0; i<ny; i++)
 {
   ylen += parameters.y_lengths[i];
   for (unsigned int j=0; j<parameters.y_subdivisions[i]; j++,my++)
     yv.push_back (parameters.y_lengths[i]/parameters.y_subdivisions[i]);
 }
      n_subdivisions.push_back (yv);
      std::vector<double> zv;
      unsigned int mz = 0;
      for (unsigned int i=0; i<parameters.n_assemblies[2]; i++)
 for (unsigned int j=0; j<parameters.z_assembly_subdivisions[i]; j++,mz++)
   zv.push_back(parameters.z_assembly_heights[i]/parameters.z_assembly_subdivisions[i]);
      n_subdivisions.push_back (zv);
  
      const Point<3> bottom_left = Point<3>();
      const Point<3> upper_right = Point<3> (xlen, ylen, t_height);

      Table<3,unsigned char> material_id(mx,my,mz);
      for (unsigned int k=0, p=0; k<parameters.n_assemblies[2]; k++)
 for (unsigned int iz=0; iz<parameters.z_assembly_subdivisions[k]; iz++,p++)
   for (unsigned int i=0, m=0; i<nx; i++)
     for (unsigned int ix=0; ix<parameters.x_subdivisions[i]; ix++,m++)
       for (unsigned int j=0, n=0; j<ny; j++)
  for (unsigned int iy=0; iy<parameters.y_subdivisions[j]; iy++,n++)
    material_id[m][n][p] = (unsigned char) (core[i][j][k]-1);
    
      GridGenerator::subdivided_hyper_rectangle (t,
       n_subdivisions,
       bottom_left,
       material_id,
       true);
      
      coremap.reinit (nx,ny);
      Triangulation<3>::raw_cell_iterator cell = t.begin_raw(),
 endc = t.end();
      for (unsigned int id=0; cell !=endc; ++cell,++id) {
 Point<3> cell_center = cell->center();
 
 double xp = 0;
 unsigned int i=0;
 for (; i<nx; i++)
   {
     if (cell_center[0]>xp && cell_center[0]<xp+parameters.x_lengths[i]) break;
     xp += parameters.x_lengths[i];
   }
 xp = 0;
 unsigned int j=0;
 for (; j<ny; j++)
   {
     if (cell_center[1]>xp && cell_center[1]<xp+parameters.y_lengths[j]) break;
     xp += parameters.y_lengths[j];
   }
 
 coremap[i][j].push_back(id);
      }
    }

  return t;
}
#endif  
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN<dim>::read_SPn_data ()
{
  for(unsigned int group = 0; group<n_groups; group++)
  {
  	std::ostringstream spn_output_filename;
  	spn_output_filename<<parameters.spn_output_file<<"-"<<group;
  	std::ifstream f_in(spn_output_filename.str().c_str());
  		
    char cbuff[20];
    std::string buffer;
    f_in>>cbuff;
   
    buffer = cbuff;
    unsigned int ndof;
    unsigned int n_moments;
    if(!buffer.compare("#Even"))
    {
     f_in>>cbuff;
     f_in>>ndof;
     f_in>>n_moments;
     dof_repitition[group].reinit(ndof,false);
     dof_neighbour_dof[group].resize(ndof);
   
     for(unsigned int moment=0; moment<n_moments; moment=moment+2)
      phi_even_spn[group][moment].reinit(ndof,false);
     for(unsigned int idof=0; idof<ndof; idof++)
     {
      for(unsigned int moment=0; moment<n_moments; moment=moment+2)
       f_in>>phi_even_spn[group][moment](idof);
      f_in.ignore(2,'E');
      
      f_in>>dof_repitition[group](idof);
      
      while(1)
      {
        f_in>>cbuff;
        buffer = cbuff;
        if(buffer.compare("#"))
          dof_neighbour_dof[group][idof].push_back(atoi(cbuff));
        else
        	 break;
      }
     }
     buffer.clear();
     cbuff[0] = '\0';
    }
    f_in>>cbuff;
    buffer = cbuff;
    if(!buffer.compare("#Odd"))
    {
     f_in>>cbuff;
     f_in>>ndof;
     f_in>>n_moments;
   
     for(unsigned int moment=1; moment<n_moments; moment=moment+2)
      phi_odd_spn[group][moment].reinit(ndof,false);
     for(unsigned int idof=0; idof<ndof; idof++)
     {
      for(unsigned int moment=1; moment<n_moments; moment=moment+2)
       f_in>>phi_odd_spn[group][moment](idof);
     }
     buffer.clear();
     cbuff[0] = '\0';
    }
    f_in.close();
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//template <int dim>
//void SN<dim>::reconstruct_SPn_psi_plus()   //Reconstruct even-parity Angular flux from SPn solution
//{
//	for(unsigned int group = 0; group<n_groups; group++)
//  {
//    std::vector<Tensor<1,dim> > J(sn_group[group]->n_dofs);
//    for(unsigned int i=0; i<J.size(); i++)
//      J[i]=0;
//      
//    std::vector<std::vector<double> > J_vector(sn_group[group]->n_dofs);
//    for(unsigned int i=0; i<J_vector.size(); i++)
//      J_vector[i].resize(2*dim, 0.0);
//      
//    std::ostringstream J_out_filename;
//	  J_out_filename << "J_vector-"<< group << ".txt";
//    std::ofstream J_out(J_out_filename.str().c_str());
//    	
//    std::vector<std::vector<double> > grad_vector(sn_group[group]->triangulation.n_active_cells());
//    for(unsigned int i=0; i<grad_vector.size(); i++)
//      grad_vector[i].resize(2*dim, 0.0);
//      
//    std::ostringstream grad_out_filename;
//	  grad_out_filename << "gradient-"<< group << ".txt";
//    std::ofstream grad_out(grad_out_filename.str().c_str());
//   
//   //=========  Buid J_recon 	
//    Quadrature<dim> dummy_quadrature (fe.get_unit_support_points());
//    FEValues<dim> fe_values (fe, dummy_quadrature, update_q_points | update_gradients  | update_JxW_values);
//      
//    Quadrature<dim-1> dummy_face_quadrature (1);
//    FEFaceValues<dim> fe_face_values (fe, dummy_face_quadrature, 
//                               update_q_points | update_normal_vectors | update_JxW_values);
//                               
//    Quadrature<dim> dummy_center_quadrature (1);
//    FEValues<dim> fe_center_values (fe, dummy_center_quadrature, update_q_points | update_gradients);
//      
//    const unsigned int   dofs_per_cell = sn_group[group]->fe.dofs_per_cell;
//    std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
//    Vector<double> phi_odd_spn_local(dofs_per_cell);
//      
//       
//    typename DoFHandler<dim>::active_cell_iterator 
//    cell = sn_group[group]->dof_handler.begin_active(),
//    endc = sn_group[group]->dof_handler.end();
//       	 
//    for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
//    { 
//       std::vector<Tensor<1,dim> > phi_odd_spn_value(dofs_per_cell);
//       for(unsigned j=0; j<dofs_per_cell; j++)
//        phi_odd_spn_value[j].clear();
//       cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
//       fe_values.reinit (cell);
//       fe_center_values.reinit(cell);
//       for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
//       {
//        phi_odd_spn_local(j) = phi_odd_spn[group][1](i_cell*dofs_per_cell+j);  //extract phi_1_spn dof values on current cell
//       }
//   //    fe_values.get_function_gradients(phi_odd_spn_local, local_dof_indices, phi_odd_spn_value);
//       const unsigned int   n_sppt_points    = dummy_quadrature.size();
//       for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
//        for (unsigned int sppt_point=0; sppt_point<n_sppt_points; ++sppt_point)  //sppt_point = local dof index
//        {
//         phi_odd_spn_value[sppt_point] += phi_odd_spn_local(j)*fe_values.shape_grad(j, sppt_point);
//        }
//                           
//       // ========= Make J_x strictly positive ============= //
////       for (unsigned int sppt_point=0; sppt_point<n_sppt_points; ++sppt_point)  //sppt_point = local dof index
////       {
////       	if(phi_odd_spn_value[sppt_point][0] < 0)  //if J_x < 0
////       		phi_odd_spn_value[sppt_point] -= 2*phi_odd_spn_value[sppt_point];  //negate sign of all component of J;
////       }
//       
//       for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
//       {
//        int i_dof = local_dof_indices[j];
//        Tensor<1,dim> temp;
//        temp = 0.0;
//        
//        bool on_face = false;    // DOF found on face, will use \vec{k} = \vec{n}, skip \vec{k} = J
//        
//        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)   // we make a loop over all the face, if we have a face where we have a boundary 
//                          // condition we go to the if
//          if (cell->at_boundary(face))
//         	 if(fe.has_support_on_face(j, face))
//         	 {
//         	   fe_face_values.reinit (cell, face);
//         	   temp += fe_face_values.normal_vector(0);
//         	   on_face = true;
//         	 }
//        
//        if(!on_face)
//        	 temp = phi_odd_spn_value[j];
//        temp /=  dof_repitition[group](i_dof);
//        J[i_dof] += temp;
//        
//        for(unsigned int component=0; component<dim; component++)
//        {
//          J_vector[i_dof][component] += fe_values.quadrature_point (j)(component)/dof_repitition[group](i_dof);
//        }
//        //J_vector[i_dof][2] += temp[0];
//        //J_vector[i_dof][3] += temp[1];
//       
//        for(unsigned int component=0; component<dim; component++)
//        {
//         grad_vector[i_cell][component+dim] += fe_center_values.shape_grad (j, 0)[component] * phi_even_spn[group][0](i_dof);  //first order quadradture, only one point on the center
//         grad_vector[i_cell][component] = fe_center_values.quadrature_point (0)[component];
//        }
//       }
//     }
//     
//     std::vector<int> zero_current_dofs;
//     for(unsigned int i_dof=0; i_dof<J.size(); i_dof++)
//     {
//       if(J[i_dof].norm() == 0)
//       	 zero_current_dofs.push_back(i_dof);
//     }
//     
//     bool pick_direction = false;   //flag indicating whether averaging will yield zero vectors
//     while(zero_current_dofs.size()>0)  //if there exist any zero current, redo the averaging step
//     {
//       std::vector<int> zero_current_dofs_temp;   //temp container for zero-current-dofs
//       for(unsigned int i=0; i<zero_current_dofs.size(); i++)
//       {
//         unsigned int i_dof = zero_current_dofs[i];
//         if(J[i_dof].norm()==0)
//         {
//           for(unsigned int nb=0; nb<dof_neighbour_dof[group][i].size(); nb++)
//         	 {	
//     //    			printf("x = %f,  ",J[dof_neighbour_dof[i_dof][nb]][0]);  //debug
//     //    			printf("y = %f,  ",J[dof_neighbour_dof[i_dof][nb]][1]);  //debug
//         		 J[i_dof] += J[dof_neighbour_dof[group][i_dof][nb]];
//         	 }
//         		  
//         	 if(J[i_dof].norm() == 0)
//         		 zero_current_dofs_temp.push_back(i_dof);
//         }
//       }
//       if(zero_current_dofs.size() == zero_current_dofs_temp.size())   //if averaging is no longer effective, switch to picking direction
//       {
//       	 pick_direction = true;
//         break;
//       } 
//       zero_current_dofs.clear();
//       zero_current_dofs = zero_current_dofs_temp;
//       zero_current_dofs_temp.clear();
//       std::cout<<"Averaging Current(J)...  # 0-J DOFs: "<<zero_current_dofs.size()<<std::endl;
//     }
//     
//     if(pick_direction)
//     {
//     	 while(zero_current_dofs.size()>0)  //if there exist any zero current, redo the averaging step
//       {
//         std::vector<int> zero_current_dofs_temp;   //temp container for zero-current-dofs
//       	 for(unsigned int i=0; i<zero_current_dofs.size(); i++)
//         {
//         	 unsigned int i_dof = zero_current_dofs[i];
//           if(J[i_dof].norm()==0)
//           {
//         	   for(unsigned int nb=0; nb<dof_neighbour_dof[group][i].size(); nb++)
//           	 {
//           	 	 unsigned int neighbour = dof_neighbour_dof[group][i_dof][nb];
//           	 	 if(J[neighbour].norm() > 0)    //pick the first no-zero neighbour as the direction
//           	  	 	J[i_dof] = J[neighbour];
//             }
//           }
//         }
//         zero_current_dofs.clear();
//         zero_current_dofs = zero_current_dofs_temp;
//         zero_current_dofs_temp.clear();
//         std::cout<<"Averaging Current(J)...  # 0-J DOFs: "<<zero_current_dofs.size()<<std::endl;
//       }
//     }
//         
//     
//     //Renormalize J_vector;
//     for(unsigned int i_dof=0; i_dof<J.size(); i_dof++)
//     {
//       if(J[i_dof].norm() != 0)
//         J[i_dof] /= J[i_dof].norm();
//       else
//         printf("Zero Current Encountered!!\n");
//     }
//     
//     for(unsigned int i_dof=0; i_dof<J_vector.size(); i_dof++)
//       for(unsigned int component=0; component<dim; component++)
//     	   J_vector[i_dof][component+dim] = J[i_dof][component];
//      
//    //After obtaining J, reconstruct the psi_plus_recon
//    for(unsigned int level=1; level<=material_data.get_n_moments()/2; level++)
//     for(unsigned int n_w=0; n_w<(dim==2?2:4)*level; n_w++)
//     {
//       unsigned int m = (dim==2?2:4)*(1+(level-1))*(level-1)/2+n_w;  //m is the direction index, Omega_m
//        
//       psi_plus_recon[group][m].reinit(sn_group[group]->dof_handler.n_dofs(),false);
//         
//       for(unsigned int i_dof=0; i_dof<psi_plus_recon[group][m].size(); i_dof++)
//       {   
//         double mu;
//         mu = Omega[m]*J[i_dof];  // mu = Omega_m * phi_1/|phi_1|
//         
//         for(unsigned int n=0;n<phi_even_spn[group].size();n=n+2)
//         {
//           double Pnm = Legendre::Pn(n, mu);
//           if (J[i_dof].norm()==0.0)  //detect whether J has a norm of zero
//           {
//           	 cout<<"norm(J) is zero!!!   mu="<<mu<<endl;
//           	 cout<<"Pn(mu) = "<<Pnm<<endl;
//           }
//           psi_plus_recon[group][m](i_dof) += (2.0*n+1.0)/(4.0*M_PI)*phi_even_spn[group][n](i_dof)*Pnm;
//           
//         }
//         
//       }
//     }
//       
//     for(unsigned int i=0; i<J_vector.size(); i++)
//     {
//       for(unsigned int component=0; component<2*dim; component++)
//         J_out<<J_vector[i][component]<<" ";
//       J_out<<endl;
//     }
//     
//     //cout<<grad_vector[i][2]*grad_vector[i][2]+grad_vector[i][3]*grad_vector[i][3]<<" ";
//  
//     for(unsigned int i=0; i<grad_vector.size(); i++)
//     {
//       for(unsigned int component=0; component<2*dim; component++)
//         grad_out<<grad_vector[i][component]<<" ";
//       grad_out<<endl;
//       //cout<<grad_vector[i][2]*grad_vector[i][2]+grad_vector[i][3]*grad_vector[i][3]<<" ";
//     }
//      
//    grad_out.close();
//    J_out.close();
//  }
//}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN<dim>::assemble_system (unsigned int group, unsigned int m)
{ // this function is used to make the the system of equations for the direct problem

 const RHS<dim> right_hand_side;              //contribution to the RHS due to even parity components
 const RHS_psi_minus<dim> right_hand_side_odd(Omega, wt);  //the odd parity counterpart
  
 QGauss<dim>  quadrature_formula(fe.degree+1); 
  FEValues<dim> fe_values (fe, quadrature_formula, 
        update_values | update_gradients | update_q_points | update_JxW_values); 
 
 const QGauss<dim-1> face_quadrature_formula(fe.degree +1);  
 const unsigned int n_face_q_points = face_quadrature_formula.size();
 FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
        update_values | update_q_points | update_normal_vectors | update_JxW_values | update_gradients);

 const unsigned int   dofs_per_cell = fe.dofs_per_cell; 
 const unsigned int   n_q_points    = quadrature_formula.size(); 
 
 FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell); 
 FullMatrix<double>   cell_scattering_matrix(dofs_per_cell, dofs_per_cell);
 Vector<double>       cell_rhs (dofs_per_cell); 
 std::vector<unsigned int> local_dof_indices (dofs_per_cell);

 std::vector<int>::iterator boundary_exist;     //we check if we have some Robin boundary condtions
 boundary_exist=std::find(parameters.boundary_conditions.begin(), parameters.boundary_conditions.end(), 2);
 bool RC_exist=false;
 
 
 typename DoFHandler<dim>::active_cell_iterator 
  cell = sn_group[group]->dof_handler.begin_active(),
  endc = sn_group[group]->dof_handler.end();

 for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
  fe_values.reinit (cell);
 
  cell_matrix = 0;
  cell_scattering_matrix = 0;
  cell_rhs = 0;
  
  double st = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
  double diffusion_coefficient = 1.0/st;   // we create the diffusion coefficient
  double ss0 = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0th-moment of the scattering cross-section
  double sa = st - ss0;  //used for manufacturered solution
  
  
  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
   for (unsigned int i=0; i<dofs_per_cell; ++i)
   {
    for (unsigned int j=0; j<dofs_per_cell; ++j)  //we give the values to the matrix
    {
     cell_matrix(i,j) += ((
          Omega[m] * fe_values.shape_grad (i, q_point) *
          Omega[m] * fe_values.shape_grad (j, q_point) +
         st*st*         
         fe_values.shape_value (i,q_point) * fe_values.shape_value (j,q_point)) *
         fe_values.JxW (q_point));
         
//     cell_scattering_matrix(i,j) += 1.0/(4.0*M_PI)*ss0*
//            fe_values.shape_value(i,q_point) *
//            fe_values.shape_value(j,q_point) *
//            fe_values.JxW(q_point);
     
     //piece-wise constant \sigma_t
     cell_matrix(i,j) += (
         st*Omega[m]* 
         ( fe_values.shape_grad (i,q_point) * fe_values.shape_value (j,q_point) +
           fe_values.shape_value (i,q_point) * fe_values.shape_grad (j,q_point)  ) *
         fe_values.JxW (q_point));

    }
  
    // we give the values for the right-hand-side
    // First, the contribution to RHS due to even parity external source
    cell_rhs(i) +=( right_hand_side.get_source (fe_values.quadrature_point (q_point), group, sa, domain_size) * 
                    Omega[m]* fe_values.shape_grad (i, q_point) + 
                    st*
                    right_hand_side.get_source (fe_values.quadrature_point (q_point), group, sa, domain_size) * 
                    fe_values.shape_value (i, q_point) )*
                    fe_values.JxW (q_point);
    // Second, the contribution to RHS due to odd parity external source      
//    cell_rhs(i) += (
//         1.0/st*right_hand_side_odd.get_source (fe_values.quadrature_point (q_point), group, domain_size, m) * 
//          Omega[m]*fe_values.shape_grad (i, q_point) *
//          fe_values.JxW (q_point) );
  }
  
  
   

   for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)   // we make a loop over all the face, if we have a face where we have a boundary 
                       // condition we go to the if
    if (cell->at_boundary(face) /*&& (parameters.boundary_conditions[cell->face(face)->boundary_indicator()] == 2)*/)
    {
     unsigned int side = cell->face(face)->boundary_indicator();
     fe_face_values.reinit (cell, face);
        
     for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) // we modify the matrix and the right-hand-side because of the boundary condition
     {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
      	//boundary term due to laplace operator
        if(Omega[m]* fe_face_values.normal_vector(q_point) > 0)  //if out-going direction, use first order equation for boundary condition
        {
          cell_rhs(i) +=  ( Omega[m] * fe_face_values.normal_vector(q_point) *
             right_hand_side.get_source (fe_face_values.quadrature_point (q_point), group, sa, domain_size)  *
             fe_face_values.shape_value(i,q_point) *
             fe_face_values.JxW(q_point));

          for (unsigned int j=0; j<dofs_per_cell; j++)  
             cell_matrix(i,j) +=  ( Omega[m] * fe_face_values.normal_vector(q_point) *
               st*
               fe_face_values.shape_value(i,q_point) *
               fe_face_values.shape_value(j,q_point)*
               fe_face_values.JxW(q_point));
               
        }
        
        //boundary term due to L* operating on source
        cell_rhs(i) +=  -( Omega[m] * fe_face_values.normal_vector(q_point) *
             right_hand_side.get_source (fe_face_values.quadrature_point (q_point), group, sa, domain_size)  *
             fe_face_values.shape_value(i,q_point) *
             fe_face_values.JxW(q_point));
        
        //piece-wise constant \sigma_t     
        for (unsigned int j=0; j<dofs_per_cell; j++)
               cell_matrix(i,j) +=  -( st*
               fe_face_values.shape_value(i,q_point) * fe_face_values.shape_value(j,q_point) *
               Omega[m]*fe_face_values.normal_vector(q_point) *
               fe_face_values.JxW(q_point));
        
          
//       if(Omega[m]* fe_face_values.normal_vector(q_point) < 0)    
//       for (unsigned int j=0; j<dofs_per_cell; j++)
//               cell_matrix(i,j) +=  -( Omega[m] * fe_face_values.shape_grad(i, q_point) *
//               fe_face_values.shape_value(i,q_point) *
//               Omega[m]*fe_face_values.normal_vector(q_point) *
//               fe_face_values.JxW(q_point));
        
      }
     }
    }
 
  cell->get_dof_indices (local_dof_indices);
  //============================== Begin ==================
//  //Compute the reconstruction residual source
//  for (unsigned int i=0; i<dofs_per_cell; ++i) // we put the matrix and the right-hand-side of the cell into the matirx and the right-hand-side of the system
//  { 
//   for (unsigned int j=0; j<dofs_per_cell; ++j)
//   {
//    sn_group[group]->system_psi_matrix.add (local_dof_indices[i],
//         local_dof_indices[j],
//         cell_matrix(i,j));
//         
//    sn_group[group]->system_scat_matrix.add (local_dof_indices[i],
//         local_dof_indices[j],
//         cell_scattering_matrix(i,j));
//   }
//  }

  //============================== End =====================

  for (unsigned int i=0; i<dofs_per_cell; ++i) // we put the matrix and the right-hand-side of the cell into the matirx and the right-hand-side of the system
  { 
   for (unsigned int j=0; j<dofs_per_cell; ++j)
    sn_group[group]->system_matrix.add (local_dof_indices[i],
         local_dof_indices[j],
         cell_matrix(i,j));

   sn_group[group]->system_rhs(local_dof_indices[i]) += cell_rhs(i);
  } 
 }

 //Add the reconstruction residual to the RHS
// if(RESIDUAL)
// {
//  Vector<double> temp;
//  temp = psi_plus_recon[group][m];
//  sn_group[group]->system_psi_matrix.vmult_add (sn_group[group]->system_rhs, temp*=(-1.0));
//  temp = phi_spn_recon[group];
//  sn_group[group]->system_scat_matrix.vmult_add (sn_group[group]->system_rhs, temp*=(1.0));
// }
 


 //we take care about the term in the right-hand-side due to the other moments
// for (unsigned int kj=0; kj < 1; kj = kj+2)  //for isotropic scattering, we are only concerned with the scalar flux: phi_even[0]
// {
//  {  
//   sn_group[group]->assemble_scattering_matrix (kj, m, (2.0*kj+1.0)/(4.0*M_PI));
//   sn_group[group]->scattering_matrix[kj].vmult_add (sn_group[group]->system_rhs, phi[group]);
//  }
// }
 

 // we take care about the hanging nodes
 sn_group[group]->hanging_node_constraints.condense (sn_group[group]->system_matrix);  
 sn_group[group]->hanging_node_constraints.condense (sn_group[group]->system_rhs);
 
  
 //Dirichlet Boundary Condition
 Quadrature<dim> dummy_quadrature (fe.get_unit_support_points());  //dummy quadrature points to contain actually the support point on unit cell
 FEValues<dim>   fe_values_dummy (fe, dummy_quadrature, update_q_points );  //auxiliary FEValues object to map the support point from unit cell to real cell

 FE_Q<dim-1> fe_face(1);
 Quadrature<dim-1> dummy_quadrature_face (fe_face.get_unit_support_points());  //dummy quadrature points to contain actually the support point on unit cell
 FEFaceValues<dim>   fe_face_values_dummy (fe, dummy_quadrature_face, update_q_points | update_normal_vectors );  //auxiliary FEValues object to map the support point from unit cell to real cell
 

 const unsigned int ndof_face_dummy = dummy_quadrature_face.size(); 
 
  cell = sn_group[group]->dof_handler.begin_active();
  endc = sn_group[group]->dof_handler.end();

 for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
 {
   fe_values_dummy.reinit (cell);
   cell->get_dof_indices (local_dof_indices);
   

     for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)   // we make a loop over all the face, if we have a face where we have a boundary 
                       // condition we go to the if
     {
   	   if (cell->at_boundary(face) /*&& (parameters.boundary_conditions[cell->face(face)->boundary_indicator()] == 2)*/)
       {
         for (unsigned int i=0; i<dofs_per_cell; ++i)
         {
           if(fe.has_support_on_face(i, face))
           {
           	 fe_face_values_dummy.reinit(cell, face);
           	 unsigned int i_face_dof;
           	 for(unsigned j=0; j<ndof_face_dummy; j++)
           	 {
           	 	 if(fe_face_values_dummy.quadrature_point(j)(0) == fe_values_dummy.quadrature_point(i)(0) &&
           	 		 fe_face_values_dummy.quadrature_point(j)(1) == fe_values_dummy.quadrature_point(i)(1) &&
           	 		 fe_face_values_dummy.quadrature_point(j)(2) == fe_values_dummy.quadrature_point(i)(2) )
           	   {
           	 		 i_face_dof =j;
           	 	 	 break;
           	 	 }
           	 }
           	 		
        	   if(Omega[m]* fe_face_values_dummy.normal_vector(i_face_dof) < 0)
        	   {
//        	   	 if((parameters.adjoint_boundary_conditions[cell->face(face)->boundary_indicator()] == 2) && (parameters.adjoint_boundary_value[cell->face(face)->boundary_indicator()] == 1))
        	       sn_group[group]->system_rhs(local_dof_indices[i]) = 1.0/(4.0*M_PI);    //Dirichlet BC value
//        	     else
//        	     	 sn_group[group]->system_rhs(local_dof_indices[i]) = 0.0/(4.0*M_PI);    //Dirichlet BC value
        	     for(unsigned i_dof=0; i_dof<sn_group[group]->dof_handler.n_dofs(); i_dof++)
        	       sn_group[group]->system_matrix.set(local_dof_indices[i],i_dof,0.0); 
        	     sn_group[group]->system_matrix.set(local_dof_indices[i],local_dof_indices[i],1.0);   //set unity on diagonal element for Dirichlet nodes
        	   }
        	   	
           }
         }
       }
     }

  }
 
  

// std::map<unsigned int,double> boundary_values; //  we use that if we some Dirichlet conditions (not converted yet)
// for (unsigned int i=0; i<parameters.boundary_conditions.size();i++)
//  if (parameters.boundary_conditions[i]==1)
//   VectorTools::interpolate_boundary_values (sn_group[group]->dof_handler,
//                                            i,
//                                            BoundaryValues<dim>(),
//                                            boundary_values);
// MatrixTools::apply_boundary_values (boundary_values,
//                                      sn_group[group]->system_matrix,
//                                      sn_group[group]->solution_moment[m],
//                                      sn_group[group]->system_rhs);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SN<dim>::compute_phi(unsigned int group)
//compute scalar flux using angluar flux
{
 Vector<double> temp;
 phi[group].reinit(sn_group[group]->dof_handler.n_dofs(),false);
 
  for(unsigned int angle = 0 ; angle < n_Omega; angle++)
  {
   temp = sn_group[group]->solution_moment[angle];
   phi[group] += temp*=((dim==2?2.0:1.0)*wt[angle]);
  }

}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//template<int dim>
//void SN<dim>::compute_reconstruct_phi()
//{
//  for(unsigned int group=0; group<n_groups; group++)
//  {
//    Vector<double> temp;
//    phi_spn_recon[group].reinit(sn_group[group]->dof_handler.n_dofs(),false);
//
//     for(unsigned int angle = 0 ; angle < n_Omega; level++)
//     {
//       temp = psi_plus_recon[group][angle];
//       phi_spn_recon[group] += temp*=((dim==2?2.0:1.0)*wt[angle]);
//     }
//  }
//}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SN<dim>::compute_response(unsigned int group)
//compute response using even parity angluar flux and adjoint boundary conditions
{
	double source = 0.0;    // \int[f \Omega*\vec{n}]
	double leakage = 0.0;   // \int[\psi+ \Omega*\vec{n}]
	response[group] = 0;  //purge the response container
	
	const RHS<dim> right_hand_side;
	  
  const QGauss<dim-1> face_quadrature_formula(fe.degree +1);  
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
         update_values | update_q_points | update_normal_vectors | update_JxW_values);
 
 
  std::vector<int>::iterator boundary_exist;     //we check if we have some Robin boundary condtions
  boundary_exist=std::find(parameters.adjoint_boundary_conditions.begin(), parameters.adjoint_boundary_conditions.end(), 2);
  bool RC_exist=false;
	
	typename DoFHandler<dim>::active_cell_iterator	//we create an iterator for the common mesh
		cell = sn_group[group]->dof_handler.begin_active(),
		endc = sn_group[group]->dof_handler.end();
	
	
	for (; cell!=endc; ++cell) 
  {
		double integral = 0;
		
		
		if (boundary_exist != parameters.adjoint_boundary_conditions.end()) 
    {
     RC_exist=true;
     for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)   // we make a loop over all the face, if we have a face where we have a boundary 
                         // condition we go to the if
      if ( cell->at_boundary(face) && (parameters.adjoint_boundary_conditions[cell->face(face)->boundary_indicator()] == 2)
      	   && (parameters.adjoint_boundary_value[cell->face(face)->boundary_indicator()] == 1) )
      {
        unsigned int side = cell->face(face)->boundary_indicator();
        fe_face_values.reinit (cell, face);
        
        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) // we modify the matrix and the right-hand-side because of the boundary condition
        {    	
          for(unsigned int angle = 0 ; angle < n_Omega; angle++)
          {
          	
            Vector<double> psi_plus_m = sn_group[group]->solution_moment[angle];
 
            std::vector<double> face_psi_values(n_face_q_points);
            std::vector<double> face_psi_values_recon(n_face_q_points);
     		    fe_face_values.get_function_values(psi_plus_m,face_psi_values);	
//   	  	    fe_face_values.get_function_values(psi_plus_recon[group][angle], face_psi_values_recon);
              
            if(Omega[angle]*fe_face_values.normal_vector(q_point) > 0.0)  //check if current Omega is out-going direction
          	{
              leakage +=  abs( Omega[angle] * fe_face_values.normal_vector(q_point) ) *
                        (face_psi_values[q_point]  // \psi(\Omega) += 2*psi+*(\Omega), \Omega*n >0  //check J_{spn} ?= J_{recon}
                             /*- face_psi_values_recon[q_point]*/)*  // \psi(\Omega) += 2*psi+*(\Omega), \Omega*n >0
                       (dim==2?2.0:1.0)*wt[angle]*                       //integrate over half the sphere                                                                                                  
                       fe_face_values.JxW(q_point);        //integrate over cell volume
            }
            else
            {             
              source +=  abs( Omega[angle] * fe_face_values.normal_vector(q_point) ) *
                       (face_psi_values[q_point])*  // \psi(\Omega) += - f(-\Omega), \Omega*n >0
                       (dim==2?2.0:1.0)*wt[angle]*                       //integrate over half the sphere                                                                                                 
                       fe_face_values.JxW(q_point);        //integrate over cell volume 
 
            }
          }
        }
      }
    }//end of boundary term		

	}
	response[group] = leakage;
  cout<<"Leakage of Interest: "<<leakage<<endl;
  cout<<"Source  of Interest: "<<source<<endl;
  cout<<"Half-Range-Current : "<<response[group]<<endl;
							
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SN<dim>::check_conservation(unsigned int group)
{
	double source = 0.0;  //total particle gain
	double sink = 0.0;    //total particle loss
	double conservation = 0.0;  //conservation = (source-sink)/source
	
	
	RHS<dim> right_hand_side;

	
	QGauss<dim>  quadrature_formula(fe.degree+1); 
  FEValues<dim> fe_values (fe, quadrature_formula, 
								update_values | update_gradients | update_q_points | update_JxW_values);	
	
	const QGauss<dim-1> face_quadrature_formula(fe.degree +1);		
	const unsigned int n_face_q_points = face_quadrature_formula.size();
	FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
								update_values | update_q_points | update_normal_vectors | update_JxW_values);

	const unsigned int   dofs_per_cell = fe.dofs_per_cell; 
	const unsigned int   n_q_points    = quadrature_formula.size(); 
 
	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell); 
	Vector<double>       cell_rhs (dofs_per_cell);	
	std::vector<unsigned int> local_dof_indices (dofs_per_cell);

	std::vector<int>::iterator boundary_exist;					//we check if we have some Robin boundary condtions
	boundary_exist=std::find(parameters.boundary_conditions.begin(), parameters.boundary_conditions.end(), 2);
	bool RC_exist=false;
	
	
	typename DoFHandler<dim>::active_cell_iterator	
		cell = sn_group[group]->dof_handler.begin_active(),
		endc = sn_group[group]->dof_handler.end();
		
	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
	  fe_values.reinit (cell);
	  
		double ss0 = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0th-moment of the scattering cross-section
		double st_origin = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
		double sa = st_origin - ss0;  //used for manufacturered solution
		
		std::vector<double> phi_values(quadrature_formula.size());
		fe_values.get_function_values(phi[group],phi_values);	
		

		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		{
			sink += phi_values[q_point]
			        *sa
			        *fe_values.JxW(q_point);		    //we calculate the absorption rate
			
			
			for(unsigned int angle = 0 ; angle < n_Omega; angle++)
      {
		  	source += right_hand_side.get_source (fe_values.quadrature_point (q_point), group, sa, domain_size) *
		  	          (dim==2?2.0:1.0)*wt[angle]*
				  		    fe_values.JxW (q_point);    //we calculate the volumetric source rate
		  }	  
		}
		

			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) 		// we make a loop over all the face, if we have a face where we have a boundary 
																												// condition we go to the if
			if (cell->at_boundary(face))
			{
				unsigned int side = cell->face(face)->boundary_indicator();
				fe_face_values.reinit (cell, face);
									
				for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)	// we modify the matrix and the right-hand-side because of the boundary condition
				{
				  for(unsigned int angle = 0 ; angle < n_Omega; angle++)
          {
	         	Vector<double> psi_plus_m = sn_group[group]->solution_moment[angle];
 
            std::vector<double> face_psi_values(n_face_q_points);
            fe_face_values.get_function_values(psi_plus_m,face_psi_values);	
     		   
     		    if(Omega[angle]*fe_face_values.normal_vector(q_point) > 0.0)  //check if current Omega is out-going direction
          	{     
              sink +=  abs( Omega[angle] * fe_face_values.normal_vector(q_point)  ) *
                     (face_psi_values[q_point]) *  // \psi(\Omega) = 2*psi+*(\Omega) - f(-\Omega), \Omega*n >0
                     (dim==2?2.0:1.0)*wt[angle]*                       //integrate over all directions                                                                                                  
                     fe_face_values.JxW(q_point);        //integrate over cell volume
            }
            else             
            {         
              source += abs( Omega[angle] * fe_face_values.normal_vector(q_point) ) *
                      (face_psi_values[q_point]) *    // \psi(\Omega) = f(\Omega), \Omega*n <0
                      (dim==2?2.0:1.0)*wt[angle]*                       //integrate over all directions                                                                                                  
                      fe_face_values.JxW(q_point);        //integrate over cell volume
            }
          }
        }
      }

		
	}
	
	cout<<"sink = "<<sink<<endl;
		cout<<"source = "<<source<<endl;
	conservation = (source-sink)/source;
  cout<<"Partical Conservation : "<<conservation<<endl;
	
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>  
void SN<dim>::output (int cycle) const
{ // this function makes the output for the scalar flux of the direct and the adjoint problem
	for(unsigned int group = 0; group<material_data.get_n_groups(); group++)
  {
    //======= Output in VTK format ==============
    DataOut<dim> data_out;
    data_out.attach_dof_handler (sn_group[group]->dof_handler);
    data_out.add_data_vector (solution[group], "solution");
   // data_out.add_data_vector (psi_plus_recon[0], "psi_plus_0");
   // data_out.add_data_vector (psi_plus_recon[1], "psi_plus_1");
   // data_out.add_data_vector (phi_even_spn[group][0], "solution_spn_recon");
    //calculate d/phi 
//    Vector<double> dPhi;
//    dPhi = solution[group];
//    if(!RESIDUAL)
//    	dPhi -= phi_even_spn[group][0];  //if using Direct Method (d\phi = \phi_Sn - \phi_SPn)
//    data_out.add_data_vector(dPhi,"dPhi");
   
    data_out.build_patches ();
    
    std::ostringstream filename;
    filename << "solution-"<< group << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output); 
    
    //===== Store d\phi in the output file
//           std::ostringstream dPhi_filename;
//           if(RESIDUAL)
//           	dPhi_filename<<"dPhi_res-"<<group<<".txt";
//           else
//           	dPhi_filename<<"dPhi_ext-"<<group<<".txt";
//   
//   				std::ofstream dPhi_out(dPhi_filename.str().c_str());
//   				
//   				
//   			std::vector<std::vector<double> > dPhi_vector(solution[group].size());
//   			for(unsigned int i=0; i<dPhi_vector.size(); i++)
//   	 			dPhi_vector[i].resize(1+dim, 0.0);
//   			Quadrature<dim> dummy_quadrature (fe.get_unit_support_points());
//   			FEValues<dim> fe_values (fe, dummy_quadrature, update_q_points | update_gradients  | update_JxW_values);
//   			
//   			const unsigned int   dofs_per_cell = sn_group[group]->fe.dofs_per_cell;
//   			std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
//   				
//   			typename DoFHandler<dim>::active_cell_iterator	
//   			cell = sn_group[group]->dof_handler.begin_active(),
//   			endc = sn_group[group]->dof_handler.end();
//   			
//   			for (unsigned int i=0; cell!=endc; ++cell, ++i) 
//   			{
//   				fe_values.reinit(cell);
//   				cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
//   				for (unsigned int j=0; j<dofs_per_cell; ++j)
//         	{
//         		unsigned int idof = local_dof_indices[j];
//            dPhi_vector[idof][dim] = dPhi(idof);
//            
//            for(unsigned int component=0; component<dim; component++) 
//              dPhi_vector[idof][component] = fe_values.quadrature_point (j)[component];
//         	}
//         }
//         for(unsigned int i=0; i<dPhi_vector.size(); i++)
//         {
//           for(unsigned int component=0; component<dim+1; component++) 
//   				   dPhi_out<<dPhi_vector[i][component]<<" ";
//   			   dPhi_out<<endl;
//   			 }
//   			
//   			dPhi_out.close();
    
    //===== Output the Response  ========
    std::ostringstream response_filename;
    response_filename<<"Response_forward-"<<group<<".txt";
    std::ofstream Response_out(response_filename.str().c_str());
    Response_out<<response[group]<<std::endl;
    Response_out.close();
   
   
    //========echo the read-in SPn data, to be checked agains input SPn data file.
//    std::ostringstream f_out_filename;
//    f_out_filename<<"out_test-"<<group;
//    std::ofstream f_out(f_out_filename.str().c_str());     //creat file to store leakage through left boundary
//    AssertThrow (f_out, ExcMessage ("Creating output file failed!"));
//    unsigned int ndof_even = dof_repitition[group].size();
//    unsigned int ndof_odd = phi_odd_spn[group][1].size();
//    unsigned int n_moments = phi_even_spn[group].size();
//    
//    f_out<<"#Even Moments "<<ndof_even<<" "<<n_moments<<endl;
//    for(unsigned int idof=0; idof<ndof_even; idof++)
//    {
//     for(unsigned int moment=0; moment<n_moments; moment=moment+2)
//      f_out<<phi_even_spn[group][moment](idof)<<" ";
//     f_out<<"E ";
//     
//     f_out<<dof_repitition[group](idof)<<" ";
//     
//     for(unsigned int i_nb=0; i_nb<dof_neighbour_dof[group][idof].size(); i_nb++)
//   		  f_out<<dof_neighbour_dof[group][idof][i_nb]<<" ";
//   		
//     f_out<<"#";
//     f_out<<endl;
//    }
//    f_out<<"#Odd Moments "<<ndof_odd<<" "<<n_moments<<endl;
//    for(unsigned int idof=0; idof<ndof_odd; idof++)
//    {
//     for(unsigned int moment=1; moment<n_moments; moment=moment+2)
//       f_out<<phi_odd_spn[group][moment](idof)<<" ";
//     f_out<<endl;
//    }
//    f_out.close();
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SN<dim>::run () 
 { // this function calls all the interesting function. Here we make all the loops over the number over the moments for the diret problem and the adjoint
 std::cout.precision(12);
 int n_refinement=parameters.n_refinement_cycles;
 double conv_tol = parameters.conv_tol;
 unsigned int dofs = 0;
 bool all_output = parameters.all_output;

// read_SPn_data();  //read-in flux moments from SPn output
// reconstruct_SPn_psi_plus(); //reconstruct psi_plus from SPn solution=
// compute_reconstruct_phi();

 
 for (unsigned int group=0; group< n_groups; group++) 
   for (unsigned int m=0; m< n_Omega; m++)
     sn_group[group]->setup_system (sn_group, m);

 for(int cycle=0; cycle<=n_refinement; cycle++) // we make this loop for each refinement 
 { 
  Timer timer;
  timer.start ();
   
  std::cout<<"Cycle = "<<cycle<<std::endl;
  
  for(unsigned int group = 0; group<n_groups; group++)   //loop through all groups
	{
    if (cycle!=0)  // here we refine the meshes except if it's the first cycle because we don't have a solution yet.
     ;//refine_grid(cycle); //right now we are not refining the grid
    else
    {
      compute_phi(group);  //get initial guess for scalar flux
      phi_old[group] = phi[group];  //make a copy of the initial guess for later DSA input
    }
  }
   
  std::vector<double> conv(n_groups, 10);
  bool not_converged = true;
  for(unsigned int j=0; not_converged && j<1 ;j++) // we solve the direct problem. we need to iterate over all the moments until we converge
  {
    not_converged = false;  //reset the not_conerged flag to FALSE for all groups
  	cout<<"Begin Source Iteration, j = "<<j<<endl;   
  	for(unsigned int group = 0; group<n_groups; group++)   //loop through all groups
   		if(conv[group] > conv_tol)
 		  {
        std::cout<<" Solving for Group #"<<group<<std::endl;
    		conv[group]=0;
    		double den=0;
    		dofs = 0;  
        for (unsigned int m=0; m< n_Omega; m++) // we make a loop over all the moments 
        { 
         //============== Solve for I+ =======================
         std::cout<<"We begin to solve the I+ equation : moment : "<<m<<std::endl;
         sn_group[group]->setup_system (sn_group, m); // we create the matrix and the right-hand-side
 
         assemble_system(group, m); // we give the values of the matrix and the right-hand-side
    
         sn_group[group]->solve(parameters.n_iteration_cg, parameters.convergence_tolerance,
                                sn_group[group]->solution_moment[m]); // we solve the system that we created just before     
         dofs = dofs + sn_group[group]->dof_handler.n_dofs();
         std::cout<<"The number of dofs is :"<<sn_group[group]->dof_handler.n_dofs()<<"\n";
        }
         
        //================ Update phi_even ================ 
        compute_phi(group);
        
        //================ DSA =========================
//        dsa->run(phi_old[group], phi[group], parameters.n_iteration_cg, parameters.convergence_tolerance, group);
        
        //============ Check Convergence ===================
        if(j!=0)
        {
          Vector<double> temp = phi[group];
          temp -= phi_old[group];
          conv[group] = temp.l2_norm();
           
          den = phi[group].l2_norm();
        } 
        else 
          conv[group]=10;
    
        conv[group] = conv[group]/den;
        cout<<"conv("<<group<<") = "<<conv[group]<<endl;  //debug
     
        phi_old[group] = phi[group];
        
        if(conv[group] > conv_tol)
         	not_converged = true;   //if any group didn't converge, repeat the source iteration process
     }
     else
     {
       std::cout<<"Group #"<<group<<" has converged!!"<<std::endl;
     }
  }
   
  
  
  for(unsigned int group = 0; group<n_groups; group++)   //loop through all groups
	{
		solution[group].reinit(sn_group[group]->n_dofs); //here we create the vector which will contain the solution and 
    
    for(unsigned int m=0; m< n_Omega; m++)
    {  
      if(dofs <= 1e5 || all_output)
        sn_group[group]->output(m); // we make the output for each moment
    }
    solution[group] = phi[group];
    compute_response(group);
    check_conservation(group);
  }
  
  if(dofs <= 1e5 || all_output)
    output (0);  // we make the output for the solution

  std::cout<<"time = "<<timer()<<std::endl<<std::endl;
 } 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int main()
{ // this function calls the others to get the parameters and call the run function
 deallog.depth_console (0);

 Timer timer_total;
 timer_total.start();
 
 std::string filename;
 filename = "project_case1_mg.prm";
  
 const unsigned int dim=Dimension;

 ParameterHandler parameter_handler;
    SN<dim>::Parameters parameters;
    parameters.declare_parameters (parameter_handler);
    parameter_handler.read_input (filename);
    parameters.get_parameters (parameter_handler);

    SN<dim> calcul (parameters);
         
  calcul.run();
 
 std::cout<<"time elapsed = "<<timer_total()<<std::endl<<std::endl;
 
 return 0;
}
