# define Dimension 3
# define AMG false
            //	double a = 7.5657e-16;   //radiation constant  (Jm-3/K-4)                     
            //	double c = 299792458;    //speed of light (m/s)  
# define sigma_Boltzmann  1.0 //5.670373e-8  //Boltzmann constant (=ac/4) double phi = 1.0;
# define k_abs_min   0
                     //0.000001
                     //0.0001     //10e-3 m.f.p's
                     //0.001        //10e-2 m.f.p's
                     //0.000001     //10e-5 m.f.p's



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
 #include <grid/grid_in.h>

   
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


   
 using namespace dealii;		
 

template <int dim> 
class MaterialData
{
	public : 
		MaterialData(const std::string& filename,
			           const Triangulation<dim>& coarse_grid,
		             const Table<2, double>&     nodal_data);

	 	double get_total_XS(const unsigned int material_id, const unsigned group) const;
		double get_moment_XS(const unsigned int material_id, const unsigned group, const unsigned int i_moment) const;
		double get_total_XS_t(const unsigned int i_cell, const unsigned i_q_point, const unsigned group) const;
		double get_T4(const unsigned int i_cell, const unsigned i_q_point) const;
		unsigned int get_n_materials () const;
		unsigned int get_n_moments () const;
		unsigned int get_n_groups () const;
		Vector<double> get_T_vertex() const;
		Vector<double> get_T4_vertex() const;
		Vector<double> get_k_abs_vertex() const;
		void set_n_moments(unsigned int N);		//set number of moments according to angular multi-grid (AMG) level
		void correct_scat_moments(unsigned int group);		//make optimal correction to the scattering moment for the current AMG level
	
	private	:
		unsigned int n_materials;
		unsigned int n_moments;
		unsigned int n_groups;
		Vector<double> temperature_vertex;
		Vector<double> T4_vertex;    //Temperature^4 for Planckian source
		Vector<double> k_abs_vertex;
		std::vector<std::vector<double> > total_XS;  		
		std::vector<Table <2, double> >  moment_XS;
	  std::vector<Table<2,  double> >  total_XS_t;
	  Table<2,  double>                T4;
};	
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
MaterialData<dim>::MaterialData(const std::string&    filename,
	                              const Triangulation<dim>&  coarse_grid,
		                            const Table<2, double>&     nodal_data)
{	// this function reads the material data
	std::ifstream f_in (filename.c_str());
	AssertThrow (f_in, ExcMessage ("Open material file failed!"));

	f_in >> n_materials;		// we read the number of media
	f_in >> n_groups;		// we read the number of groups
	f_in >> n_moments;		// we read the number of moments
		
	f_in.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
		
  DoFHandler<dim> dof_handler (coarse_grid);
  FE_Q<dim> fe(1);
  dof_handler.distribute_dofs (fe);
    
  QGauss<dim>  quadrature_formula(2*fe.degree); 
  unsigned int   n_q_points    = quadrature_formula.size(); 
  FEValues<dim> fe_values (fe, quadrature_formula, 
        update_values | update_q_points | update_JxW_values); 
	
	//resize XS containers
  total_XS.resize(n_groups);
  moment_XS.resize(n_groups);
  total_XS_t.resize(n_groups);
  temperature_vertex.reinit(dof_handler.n_dofs());
  T4_vertex.reinit(dof_handler.n_dofs());   
	k_abs_vertex.reinit(dof_handler.n_dofs());
  T4.reinit(coarse_grid.n_active_cells(), n_q_points);
  for(unsigned int g=0; g<n_groups; g++)
  {
    total_XS[g].resize(n_materials);
    moment_XS[g].reinit(n_materials, n_moments);
    total_XS_t[g].reinit(coarse_grid.n_active_cells(), n_q_points);
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
	
	
	for(unsigned int i_dof = 0; i_dof < dof_handler.n_dofs(); i_dof++)
	{
		temperature_vertex(i_dof) = nodal_data(i_dof, 0);          
		T4_vertex(i_dof) = pow(temperature_vertex(i_dof), 4.0);
		k_abs_vertex(i_dof) = std::fabs(nodal_data(i_dof, 1)); //*100;  //debug, change unit from 1/m to 1/cm
	}
	
  
  // Compute T-dependent total XS for each cell and each quadrature point      
  typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

  for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
    fe_values.reinit (cell);
     	
    for(unsigned int g=0; g<n_groups; g++)
    {
     	//Evaluate local temperature based on trilinear model
   	  std::vector<double> local_T4(n_q_points);
   	  std::vector<double> local_k_abs(n_q_points);
   		fe_values.get_function_values(T4_vertex, local_T4);
   		fe_values.get_function_values(k_abs_vertex, local_k_abs);
   			
   		Vector<double> st_t(n_q_points);   //T-dependent total XS
   		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
   		{
   			if(g==0)
   			  T4(i_cell, q_point) = local_T4[q_point];
   		  total_XS_t[g](i_cell, q_point) = local_k_abs[q_point];
   		}
    }
  }
  
  dof_handler.clear();
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
template <int dim>
double MaterialData<dim>::get_total_XS(const unsigned int material_id, const unsigned int group) const
{
 return total_XS[group][material_id]; 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
template <int dim>
double MaterialData<dim>::get_moment_XS(const unsigned int material_id, const unsigned int group, const unsigned int i_moment) const
{
 return moment_XS[group][material_id][i_moment]; 
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
template <int dim>
double MaterialData<dim>::get_total_XS_t(const unsigned int i_cell, const unsigned i_q_point, const unsigned group) const
{
	return total_XS_t[group][i_cell][i_q_point];
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
template <int dim>
double MaterialData<dim>::get_T4(const unsigned int i_cell, const unsigned i_q_point) const
{
	return T4[i_cell][i_q_point];
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
unsigned int MaterialData<dim>::get_n_materials () const
{
	return n_materials;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
unsigned int MaterialData<dim>::get_n_moments () const
{
	return n_moments;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void MaterialData<dim>::set_n_moments(unsigned int N)		//set SPn order in different level of AMG
{
	n_moments = N;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
unsigned int MaterialData<dim>::get_n_groups () const
{
 return n_groups;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Vector<double> MaterialData<dim>::get_T_vertex () const
{
 return temperature_vertex;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Vector<double> MaterialData<dim>::get_T4_vertex () const
{
 return T4_vertex;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Vector<double> MaterialData<dim>::get_k_abs_vertex () const
{
 return k_abs_vertex;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void MaterialData<dim>::correct_scat_moments(unsigned int group)		//get the optimal corrected cross-section for current AMG level. 
																						//Although the function is written in mult-group fashion,
																						//It actually only considers 1-group scenario thus only valid for 1-group
{
	std::cout<<"Correcting XS for Group #"<<group<<std::endl;	
	for (unsigned int i=0; i<n_materials; i++)
	{
		double correct_moment_XS = (moment_XS[group][i][n_moments/2]+moment_XS[group][i][n_moments-1])/2.0;
		for (unsigned int j=0; j< n_moments; j++)
		{
			moment_XS[group][i][j]=moment_XS[group][i][j] - correct_moment_XS;

			
std::cout<<moment_XS[group][i][j]<<"  ";    //debug				
		}
std::cout<<endl;
		total_XS[group][i] = total_XS[group][i] - correct_moment_XS;
	}
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
{	// we use this function when we have some Dirichlet boundary condition
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
		void set_suppress(bool suppressed);
		virtual double get_source (const Point<dim>   &p, unsigned int group, const double T4, bool diffusion, const unsigned int  component = 0) const;
		virtual double get_Jinc (const Point<dim>   &p, double value, unsigned int group, unsigned int m, std::vector<double> mu, std::vector<double> wt, const unsigned int  component = 0 ) const;
	private :
		bool suppress;  //To suppress the external source in all AMG levels but the top level. default to be false
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double RHS<dim>::get_source (const Point<dim> &p, unsigned int group, const double T4, bool diffusion, const unsigned int /*component*/) const 
 {	// we give the source for the direct problem
	double value = 0;


	
	//if(p[0] <= 2  && p[1] >=9 && p[1] <= 11 && p[2] >=9 && p[2] <= 11)  //3-D
	//	value=1;						    																					//3-D
	
//	if(p[0] <= 2  && p[1] >=9 && p[1] <= 11)   //2-D
//		value=1;																 //2-D
	
	//if(p[0] <= 2)			//1-D
	//	value=1;				//1-D
	
//	if(p[0] <= 2  && p[1] >=9 && p[1] <= 11)
//		value=1;  
	
//	double a = 7.5657e-16;   //radiation constant  (Jm-3/K-4)                     
//	double c = 299792458;    //speed of light (m/s)                     	
// 	double sigma_Boltzmann = 5.670373e-8;  //Boltzmann constant (=ac/4) double phi = 1.0;
 	
	if(diffusion)
	{
		value = 4.0*sigma_Boltzmann*T4;
	}
	else
	{
		//value = phi;
		//value = 4.0*sigma_Boltzmann*T4/(4.0*M_PI);
		value = 4.0*sigma_Boltzmann*T4;
	}	
	if(suppress)
		value = 0.0;   //source suppressed in AMG coarse grid
		
	return value; 
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
double RHS<dim>::get_Jinc (const Point<dim> &p, double value, unsigned int group, unsigned int m, std::vector<double> mu, std::vector<double> wt, const unsigned int /*component*/) const 
 {	// we use that function when we have en incoming current to retrun the value
   double return_value;
   
   double alpha = 0;    //renormalization factor
 	 double half_range_integral = value/2*0.5;        //actual half range current, auxiliary variable to compute \alpha
 	 double half_range_sum = 0;   //quadrature half range current, auxiliary variable to compute \alpha
 	 for(unsigned i=0; i<wt.size()/2; i++)
 	   half_range_sum += value/2*mu[i]*wt[i]*2;
 	   
// 	 if(half_range_sum!=0.0)
// 	   alpha = half_range_integral/half_range_sum;
// 	 else
 	 	 alpha = 1.0;
// 	 cout<<"alpha = "<<alpha<<endl;   //debug
   
   if (suppress == false)
   		return_value = (value)*alpha;
   else
   		return_value = 0.0;
   
   return_value = return_value;

   return return_value;  
   
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void RHS<dim>::set_suppress(bool suppressed)
{
	suppress = suppressed;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
class RHS_psi_minus : public Function<dim>		//this is the RHS due to odd parity components
	{
	public :
		RHS_psi_minus(std::vector<double> mu, std::vector<double> wt);
	  Tensor<1,dim> get_vector_source (const Point<dim>   &p, unsigned int group, std::vector<double> domain_size, const unsigned int  m = 0) const;
	  void set_suppress(bool suppressed);
	private :
		std::vector<double> mu;
		std::vector<double> wt;
		bool suppress;  //default to be false
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
RHS_psi_minus<dim>::RHS_psi_minus(std::vector<double> mu, std::vector<double> wt)
	:
		mu(mu),
		wt(wt)
		{
		}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
Tensor<1,dim> RHS_psi_minus<dim>::get_vector_source (const Point<dim> &p, unsigned int group,	std::vector<double> domain_size, const unsigned int m) const 
 {	// we give the source for the direct problem
        Tensor<1,dim> value(true);
        
        double phi = 1.0;
        double xlen = 20;
//        value[0] = mu[m]*phi*2.0*M_PI/xlen*cos(2.0*M_PI*(p[0]-xlen/2.0)/xlen);  //sin(x) manufacturered solution
//        value[0] = mu[m]*phi*2.0*M_PI/xlen*cos(2.0*M_PI*(p[0]-xlen/2.0)/xlen)*sin(2.0*M_PI*(p[1]-xlen/2.0)/xlen);
//        value[1] = mu[m]*phi*2.0*M_PI/xlen*sin(2.0*M_PI*(p[0]-xlen/2.0)/xlen)*cos(2.0*M_PI*(p[1]-xlen/2.0)/xlen);  //sin(x)sin(y) manufacturered solution

			if(suppress)
				Tensor<1,dim> value(true);  //suppressed in AMG coarse grid
//=======================

	// if((p[0]<=10&&(p[1]<=10||p[1]>=20))||(p[0]>=20&&(p[1]<=10||p[1]>=20)))
		// value =1;
	// if((p[0]<=20&&p[0]>=10&&p[1]<=20&&p[1]>=10))
		// value=1;
	
	//if(p[0] <= 2  && p[1] >=9 && p[1] <= 11 && p[2] >=9 && p[2] <= 11)  //3-D
	//	value=1;						    //3-D
	
	return value; 
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void RHS_psi_minus<dim>::set_suppress(bool suppressed)
	//used in Angular Multi-grid mode, to suppress the external source on all levels but the top level
{
	suppress = suppressed;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>												
class SPN_group
{
	public :
		SPN_group (const unsigned int        		m,		//m is the direction index
					MaterialData<dim>       		&material_data_new,
          MaterialData<dim>					&material_data_old,
					const Triangulation<dim>		&coarse_grid,
					const FiniteElement<dim> 		&fe,
					const std::vector<std::vector<double> >        &Pnm);
        ~SPN_group();			
		
		void matrix_reinit();			// we reinitialize the matrices that we use
		void setup_system (const std::vector<SPN_group<dim>*> SPN_group, unsigned int m); // we create the pattern of the matrices that we use
		void assemble_scattering_matrix (const unsigned int n, const unsigned int m, double coefficient);		// we assemble the scattering mass matrix for Pn scattering

		void solve (unsigned int n_iteration_cg, double convergence_tolerance, Vector<double> &solution_moment);	// we solve the system of equation that we have
		void output (int cycle, unsigned int m) const;									// we make the output for one direction for the direct and the adjoint problem				
		
		Triangulation<dim>				triangulation; 
		const FiniteElement<dim>	  	&fe;
		DoFHandler<dim>					dof_handler;		
		ConstraintMatrix     			hanging_node_constraints;
		std::vector<Vector<double> >					solution_moment;    //one solution_moment[m] Vector for each quadrature direction
		Vector<double>					system_rhs;
		SparsityPattern					sparsity_pattern;
		SparseMatrix<double>			system_matrix;	
		std::vector<SparseMatrix<double> > 		scattering_matrix;  
		MappingQ1<dim>					mapping;
		unsigned int              n_dofs;

		Vector<float> estimated_error_per_cell, adjoint_estimated_error_per_cell, new_estimated_error_per_cell;
	
	
	private :
		unsigned int							group;
		unsigned int              n_moments;  //number of moments(directions) in current AMG level
		unsigned int              n_groups;
		MaterialData<dim>           			&material_data_new;
		MaterialData<dim>								&material_data_old;	
		const std::vector<std::vector<double> >         &Pnm;  //Array storing value of Legendre polynomial of order(n) at mu(m);
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SPN_group<dim>::SPN_group(const unsigned int        		group,		// this is the constructor of the SPN_group object, we give some values by default
                            MaterialData<dim>       		&material_data_new,
                            MaterialData<dim>					&material_data_old,
                            const Triangulation<dim> 		&coarse_grid,
                            const FiniteElement<dim> 		&fe,
                            const std::vector<std::vector<double> >        &Pnm)
                 :
				 fe (fe),
				 dof_handler (triangulation),
                 group (group),
                 material_data_new (material_data_new),
                 material_data_old (material_data_old),
		 Pnm(Pnm)
				
 {
   triangulation.copy_triangulation (coarse_grid);
	 n_groups = material_data_old.get_n_groups(),
	 n_moments = material_data_new.get_n_moments();
   dof_handler.distribute_dofs (fe);
   n_dofs = dof_handler.n_dofs(),
   solution_moment.resize(n_moments/2);
   for(unsigned int m=0; m<n_moments/2; m++)
     solution_moment[m].reinit(n_dofs, false); 
     
   scattering_matrix.resize(n_moments);
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SPN_group<dim>::~SPN_group()
{// this is the destructor of the SPN_group object
  dof_handler.clear(); 
  cout<<"SPN_group destructed"<<endl;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN_group<dim>::assemble_scattering_matrix (const unsigned int n, const unsigned int m, double coefficient)			
{	// this function is used to assemble the mass matrix that accounts for the even-parity scattering source																	
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
							local_mass_matrix(i,j) += (	
													coefficient*material_data_old.get_moment_XS(cell->material_id(), group, n)*Pnm[n][m] *
													fe_values.shape_value(i,q_point) *
													fe_values.shape_value(j,q_point) *
													fe_values.JxW(q_point));
													
			const unsigned int dofs_per_cell = fe.dofs_per_cell;
			std::vector<unsigned int> local_dof_indices (dofs_per_cell);
			cell->get_dof_indices (local_dof_indices);
												
			for (unsigned int i=0; i<dofs_per_cell; ++i)			// we add the value of the celle in the global matrix. We still must multiply this matrix by the solution
				for (unsigned int j=0; j<dofs_per_cell; ++j)		// for the associated moment. We do that in assemble_system and assemble_adjoint_system
					scattering_matrix[n].add (local_dof_indices[i], // pour avoir le membre de droite, il faut encore multiplier la matrice globale par 
							  local_dof_indices[j],				
							  local_mass_matrix(i,j));		
		}	
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN_group<dim>::setup_system (const std::vector<SPN_group<dim>*> SPN_group, unsigned int m)
{	// here we give the pattern of all the matrices and the right-hand-side that we use 
	dof_handler.distribute_dofs (fe);
		
	sparsity_pattern.reinit (n_dofs,
                           n_dofs,
                           dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern); // we create the pattern of the matrix
	    

    solution_moment[m].reinit (n_dofs, false); //we create the pattern of the solution
	
    system_rhs.reinit (n_dofs); //we create the pattern of the right-hand-side
	
	hanging_node_constraints.clear ();						// here we take care about the hanging nodes
	DoFTools::make_hanging_node_constraints (dof_handler,
                                            hanging_node_constraints);

	hanging_node_constraints.close ();
	hanging_node_constraints.condense (sparsity_pattern);
											
    sparsity_pattern.compress(); 
 
    system_matrix.reinit (sparsity_pattern); // now the matrix receive her pattern
	
	scattering_matrix.resize (n_moments);
	
	for (unsigned int moment=0; moment<n_moments; moment++)
			scattering_matrix[moment].reinit (sparsity_pattern);	
 }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN_group<dim>::matrix_reinit()
{ // we just reinitialize the all the matrix and the right-hand-side that we use in the code
	system_rhs.reinit (n_dofs);
	system_matrix.reinit (sparsity_pattern);
	for (unsigned int m=0; m<n_moments; m++)
			scattering_matrix[m].reinit (sparsity_pattern);	
} 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
template <int dim>
void SPN_group<dim>::solve (unsigned int n_iteration_cg, double convergence_tolerance, Vector<double> &solution_moment)
{	// here we just solve the system of equation thanks to CG we use a SSOR preconditioner 
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
void SPN_group<dim>::output (int cycle, unsigned int m) const
 {	//make the output for each direction  for the direct and the adjoint problem
	DataOut<dim> data_out;
 
	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution_moment[m], "solution");
 
	data_out.build_patches ();
 
	std::ostringstream filename;
	
	filename << "solution_direction-" << m << "-cycle-" << cycle << ".vtk";
 
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
		DSA (	const MaterialData<dim>       		&material_data,
					const std::vector<int>			boundary_conditions,
					const std::vector<double>		boundary_value,
					const Triangulation<dim>		&coarse_grid,
					const FiniteElement<dim> 		&fe,
          const std::vector<double>		&mu,
					const std::vector<double>		&wt);
    ~DSA();			
		
		void matrix_reinit();			// we reinitialize the matrices that we use, PROBABLY NOT NEEDED
		void setup_system (); // we create the pattern of the matrices that we use
		void assemble_system(Vector<double> &phi_old, Vector<double> &phi, Vector<double> &J_old, Vector<double> &J, unsigned int group);
		void run(Vector<double> &phi_old, Vector<double> &phi, Vector<double> &J_old, Vector<double> &J, 
		         unsigned int n_iteration_cg, double convergence_tolerance, unsigned int group);

		void solve (unsigned int n_iteration_cg, double convergence_tolerance);	// we solve the system of equation that we have
																																						// the solution is correction to scalar flux, phi
		void compute_dJ(Vector<double> &J_old, Vector<double> &J, unsigned int group);  //we solve for the correction for J;
		
		Triangulation<dim>				triangulation; 
		const FiniteElement<dim>	  	&fe;
		DoFHandler<dim>					dof_handler;		
		ConstraintMatrix     		hanging_node_constraints;
		Vector<double>					solution_dphi; //correction to phi
		Vector<double>					dJ;		//correction to J
		Vector<double>					system_rhs;
		SparsityPattern					sparsity_pattern;
		SparseMatrix<double>			system_matrix; 
		SparseMatrix<double>			system_rhs_matrix; 
		MappingQ1<dim>					mapping;

	
	private :
		const MaterialData<dim>          &material_data;
		std::vector<int>						boundary_conditions;
		std::vector<double>					boundary_value;
		const std::vector<double>					&mu;
		const std::vector<double>					&wt;
};
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
DSA<dim>::DSA(// this is the constructor of the DSA object, we give some values by default
                            const MaterialData<dim>       		&material_data,
                            const std::vector<int>			boundary_conditions,
														const std::vector<double>		boundary_value,
                            const Triangulation<dim> 		&coarse_grid,
                            const FiniteElement<dim> 		&fe,
                            const std::vector<double>		&mu,
														const std::vector<double>		&wt)
                 :
				 fe (fe),
				 dof_handler (triangulation),
				 material_data (material_data),
				 boundary_conditions(boundary_conditions),
				 boundary_value(boundary_value),				 
                 mu(mu),
                 wt(wt)				
 {
   triangulation.copy_triangulation (coarse_grid);
   dof_handler.distribute_dofs (fe);
   solution_dphi.reinit(dof_handler.n_dofs(),false);
   dJ.reinit(triangulation.n_active_cells()*fe.dofs_per_cell,false);
 }
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
DSA<dim>::~DSA()
{// this is the destructor of the SPN_moment object
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
{	// here we give the pattern of all the matrices and the right-hand-side that we use 
	dof_handler.distribute_dofs (fe);
		
	sparsity_pattern.reinit (dof_handler.n_dofs(),
                           dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern); // we create the pattern of the matrix
	    
    solution_dphi.reinit (dof_handler.n_dofs()); //we create the pattern of the solution
	
    system_rhs.reinit (dof_handler.n_dofs()); //we create the pattern of the right-hand-side
	
	hanging_node_constraints.clear ();						// here we take care about the hanging nodes
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
void DSA<dim>::assemble_system (Vector<double> &phi_old, Vector<double> &phi, Vector<double> &J_old, Vector<double> &J, unsigned int group)
{	// this function is used to make the the system of equations for the direct problem
	
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

	std::vector<int>::iterator boundary_exist;					//we check if we have some Robin boundary condtions
	boundary_exist=std::find(boundary_conditions.begin(), boundary_conditions.end(), 2);
	bool RC_exist=false;
	
	Vector<double> dphi = phi;		//dphi = phi - phi_old, a temporary container, will appear near the end of this function
	
	double mu_bar = 0.0;		// <mu>,  used in the boundary condition
	for (unsigned int m=0; m<mu.size()/2; m++)
		mu_bar += mu[m]*wt[m]*2.0;
	if(mu.size()==1)
		mu_bar = 0.5;
	
	
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
		double ss0 = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0th-moment of the scattering cross-section
		std::vector<double> st_t(n_q_points);
			std::vector<double> sa_t(n_q_points);
			std::vector<double> T4(n_q_points);
			std::vector<double> diffusion_coefficient(n_q_points);
			double st_t_zero = k_abs_min;    //1e-5 m.f.p.
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
			{
			  st_t[q_point] = material_data.get_total_XS_t(i_cell, q_point, group);
		    sa_t[q_point] = material_data.get_total_XS_t(i_cell, q_point, group) - ss0;
				  diffusion_coefficient[q_point] = 1.0/(3.0*std::max(st_t[q_point], st_t_zero));   // we create the diffusion coefficient
		    T4[q_point] = material_data.get_T4(i_cell, q_point);
		  }
		double ss1, ss1_old;
		if (material_data.get_n_moments()>1)
		{
			ss1 = material_data.get_moment_XS(cell->material_id(), group, 1); //get 1st-moment of the scattering cross-section
			ss1_old = material_data.get_moment_XS(cell->material_id(), group, 1); //get 1st-moment of the scattering cross-section from previous grid
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		  	diffusion_coefficient[q_point] = 1.0/(3.0*(std::max(st_t[q_point], st_t_zero)-ss1) );   // we create the diffusion coefficient
		}
		
		  
		double sa = st - ss0;  //calculate the absorption cross-section
		
		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for (unsigned int j=0; j<dofs_per_cell; ++j)		//we give the values to the matrix
				{
					cell_matrix(i,j) += ((diffusion_coefficient[q_point] *
									fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) +
									sa_t[q_point] *									
									fe_values.shape_value (i,q_point) * fe_values.shape_value (j,q_point)) *
									fe_values.JxW (q_point));

					// assembly the matrix for the 0th moment scattering residual on the RHS
			  	cell_rhs_matrix(i,j) += (ss0*
			  					fe_values.shape_value (i, q_point) * fe_values.shape_value (j, q_point) * 
			  					fe_values.JxW (q_point) );
			  					
			  	// assembly the matrix for the 1st moment scattering residual on the RHS				
			  	if (material_data.get_n_moments()>1)
			  		cell_rhs(i) += ( ss1_old/(std::max(st_t[q_point], st_t_zero)-ss1)*
			  	  				( J(i_cell*dofs_per_cell + j) - J_old(i_cell*dofs_per_cell + j) )*
			  						fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) * 
			  						fe_values.JxW (q_point) );
			  }
			}	   
			

		//we put the Robin boundary conditions if they exist
		if (boundary_exist != boundary_conditions.end()) 
		{
			RC_exist=true;
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) 		// we make a loop over all the face, if we have a face where we have a boundary 
																							// condition we go to the if
      {
      	unsigned int side = cell->face(face)->boundary_indicator()-1;
				if (cell->at_boundary(face) && (boundary_conditions[side] == 2))
				{
					fe_face_values.reinit (cell, face);
								
					for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)	// we modify the matrix because of the boundary condition
																																						// RHS remains unchanged because we can only have zero boundary source in DSA
					{
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
		}
 
		cell->get_dof_indices (local_dof_indices);

		for (unsigned int i=0; i<dofs_per_cell; ++i)	// we put the matrix and the right-hand-side of the cell into the matirx and the right-hand-side of the system
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
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
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
{	// here we just solve the system of equation thanks to CG we use a SSOR preconditioner 
	SolverControl           solver_control (n_iteration_cg, convergence_tolerance);
	SolverCG<>              cg (solver_control);
    
	PreconditionSSOR<> preconditioner; 
	preconditioner.initialize(system_matrix, 1.2);  
	cg.solve (system_matrix, solution_dphi, system_rhs,
            preconditioner);	

	hanging_node_constraints.distribute(solution_dphi);	
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void DSA<dim>::compute_dJ(Vector<double> &J_old, Vector<double> &J, unsigned int group)
	//compute correction to J after solving for correction for phi
{
	double temp;
		
		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
		typename DoFHandler<dim>::active_cell_iterator	
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
		
		for (unsigned int i=0; cell!=endc; ++cell, ++i) 
		{
			double st = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
			double ss1 = material_data.get_moment_XS(cell->material_id(), group, 1);  //get the scattering moment cross-section
			double ss1_old = material_data.get_moment_XS(cell->material_id(), group, 1); //get 1st-moment of the scattering cross-section from previous grid

			cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
			
			for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
			{
				temp = 0.0;  //reset temporary container for phi_odd[moment][n], where n is the discontinuous basis function index
				temp += -1.0/(3.0*(st-ss1))*solution_dphi(local_dof_indices[j]);  //grad(phi^{l+1/2}) term
				temp += ss1_old/(st-ss1)*( J(i*dofs_per_cell+j) - J_old(i*dofs_per_cell+j) );  //J - J_old term
				dJ(i*dofs_per_cell+j) = temp;  //temp is saved in the same way as phi_odd[moment] value at position (i*dofs_per_cell+j);
																			// we only save dJ values at Discontinuous DOFs.
			}
		}

}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void DSA<dim>::run(Vector<double> &phi_old, Vector<double> &phi,
									 Vector<double> &J_old, Vector<double> &J,
									 unsigned int n_iteration_cg, double convergence_tolerance, unsigned int group )
{
	std::cout<<"We begin to solve the DSA equation: "<<std::endl;
	setup_system (); // we create the matrix and the right-hand-side 
	assemble_system(phi_old, phi, J_old, J, group); // we give the values of the matrix and the right-hand-side

	solve(n_iteration_cg, convergence_tolerance); // we solve the system that we created just before			
	phi += solution_dphi;  //add the DSA correction to the scalar flux
	
	if(material_data.get_n_moments()>1)
	{
		compute_dJ(J_old, J, group);
		J += dJ;  //add the P1SA correction to the current;	
	}
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
class SPN // here we define the SPN class which contains the Parameters class
{
	public :
	
		class Parameters 
			{
			public:
				Parameters ();
	
				static void declare_parameters (ParameterHandler &prm);
				void get_parameters (ParameterHandler &prm);

				unsigned int pdim;					  // dimension of the problem
				bool go;							  // true if we have to make a goal oriented calculation
				bool all_output;
				bool current;
				unsigned int n_points;
				std::vector<double> point;	
				unsigned int n_refinement_cycles;     //number of refinement cycles
				unsigned int refinement_level_difference_limit; // limit for refinement levels on different meshes
				unsigned int n_iteration_cg;		  	  //maximum number of iterations for CG
				double convergence_tolerance;	  //criterion of convergence fo CG
				double conv_tol;				// critere de convergence sur les moments
				unsigned int fe_degree;				 // degree of the finite elements
		
				std::string geometry_input_file;
				std::string assembly_file;
				std::string material_data_file;
				std::string output_file;
				std::string spn_output_file;

				std::vector<unsigned int> n_assemblies;		//number of assemblies
				std::vector<int> boundary_conditions;		//vector to know which kind of boundary conditions we have for the direct problem
				std::vector<double> boundary_value;			// value of the bondary if we have some Robin conditions
				std::vector<int> adjoint_boundary_conditions;		//vector to know which kind of boundary conditions we have for the adjoint problem
				std::vector<double> adjoint_boundary_value;			// value of the bondary if we have some Robin conditions

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
		
		SPN(Parameters &prm);
		~SPN();
		void run ();
		Parameters                   	&parameters;
		
	
	private :
		
		void assemble_system (unsigned int moment, unsigned int group, bool suppress_source); // make the system of equations for the direct problem
		void get_common_mesh(int cycle);		//give us the mesh which is the union of all the meshes of the different moments
		void output (int cycle) const;			// we make the output of the solution
  	void compute_phi_even(unsigned int group, unsigned int n_moments);
  	void compute_phi_odd(unsigned int group, unsigned int n_moments, bool suppress_source);
  	void build_q_minus(unsigned int group, unsigned int moment);  //project the q-minus onto the finite element basis function of dB(x)/dx
  	void compute_angular_quadrature(unsigned int n_moments, 
																					std::vector<std::vector<double> > &Pnm,
																					std::vector<double> &mu,
																					std::vector<double> &wt);
		void reinit_phi_even_odd(unsigned int group);   //invoked at beginning of each AMG cycle to initialize the phi_even/odd container
		void attach_neighbour_dofs(unsigned int idof, unsigned int neighbour_idof, unsigned int group);  //attach the neighbour_idof to the neighbour list of idof
		void build_dof_repitition_neigbourhood(unsigned int group);  //count the repitition of each dof and neighbouring dofs associated with this one
		void compute_Leakage(unsigned int group);                            //compute leakage through the given boundary
		void check_conservation(unsigned int group);
		
		Table<2,std::vector<unsigned int> >  	coremap;
		const Triangulation<dim>        		coarse_grid;
		const Table<2, double>              nodal_data;    //nodal data from gray diffusion solution
		
	  const MaterialData<dim>            				material_data;
		MaterialData<dim>										material_data_new;
		MaterialData<dim>										material_data_old;  //auxiliary material_data used in Angular Multi-Grid
																												//contains n_moments and scat_XS from previous grid
		
		FE_Q<dim>						fe;
		Triangulation<dim> 				initial_grid();
		Table<2, double>            read_nodal_data();
		std::vector<double>				domain_size;		//dimension of the problem along x,y,z;
		Table<2, Vector<double> >						q_minus;         //odd parity external source (q-) projected onto the basis function of \grad B(r)
		std::vector<Vector<double> >						solution;
		Table<2, Vector<double> >		phi_even;  //even Legendre moments of the angular flux
		Table<2, Vector<double> >		phi_odd;   //odd Legendre moments of the angular flux
		Table<2, Vector<double> >		phi_even_old;  //AMG auxiliary variable to store temp phi
		Table<2, Vector<double> >		phi_odd_old;   //........
		Table<2, Vector<double> >		phi_even_new;  //........
		Table<2, Vector<double> >		phi_odd_new;   //........
		Table<2, Vector<double> >		phi_even_previous;  //auxiliary variable to compute convergence,
		Table<2, Vector<double> >		phi_odd_previous;   //to store phi_even/odd from previous iteration
		std::vector<double>         response;
			
	  std::vector<Vector<double> >									phi_old;   //auxiliary phi_0 used in DSA
	  std::vector<Vector<double> >									J_old;   	 //auxiliary phi_1 used in P1SA
	  std::vector<Vector<double> >									dof_repitition;  //Shared DOF repitition counter
	  std::vector<std::vector<std::vector<unsigned int> > > dof_neighbour_dof;
 		std::vector<std::vector<double> >        Pnm;  //Array storing value of Legendre polynomial of order(n) at mu(m);
		std::vector<double>                  mu;  //get the value for \mu's according to the order of Sn expansion to be used.
		std::vector<double>									wt;  //weight corresponding to \mu

		std::ofstream             		outp;
		Triangulation<dim>				common_triangulation;
//		DoFHandler<dim>*				common_dof_handler;
		MappingQ1<dim>					common_mapping;
	
		std::vector<SPN_group<dim>*> 			spn_group;
	  DSA<dim>*			dsa;
};	
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SPN<dim>::Parameters::Parameters()		// this is the constructor for the parameters objects, we give some value by default 
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
void SPN<dim>::Parameters::declare_parameters (ParameterHandler &prm)
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
		     Patterns::List (Patterns::Integer(0)),
		     "List of boundary conditions for each of the 2*dim "
		     "faces of the domain");
  prm.declare_entry ("Boundary values", "",
		     Patterns::List (Patterns::Double()),
		     "List of boundary values for each of the 2*dim "
		     "faces of the domain");
  prm.declare_entry ("Adjoint boundary conditions", (dim == 2? "2,2,2,2" : "2,2,2,2,2,2"),
		     Patterns::List (Patterns::Integer(0)),
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
void SPN<dim>::Parameters::get_parameters (ParameterHandler &prm)
{ // here we read the project.prm and we give the good value for all the parameters
  pdim                    = prm.get_integer ("Dimension");
  AssertThrow (pdim == dim, 
	       ExcMessage ("Problem dimension is not consistent with the code!"));
  go                          = prm.get_bool    ("Goal oriented");
  current 					  = prm.get_bool    ("Current");
  n_points					  = prm.get_integer ("Number of interesting points");
  all_output                  = prm.get_bool    ("All output");
  n_refinement_cycles         = prm.get_integer ("Refinement cycles");
  refinement_level_difference_limit      = prm.get_integer ("Refinement level difference limit");
  fe_degree                   = prm.get_integer ("Finite element degree");
  convergence_tolerance		  = prm.get_double  ("Convergence tolerance");
  conv_tol          		  = prm.get_double  ("Convergence tolerance for the moments");
  n_iteration_cg	     	  = prm.get_integer ("Maximum number of CG iterations");
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
SPN<dim>::SPN(Parameters &prm)  // constructor of the SPN object 
			:
			parameters (prm),
			      coarse_grid(initial_grid()),
            nodal_data(read_nodal_data()),
            material_data (parameters.material_data_file, coarse_grid, nodal_data),
            material_data_new (material_data),  //save material_data from previous grid into material_data_old
            material_data_old (material_data),
            fe (parameters.fe_degree),       //scalar FE for psi+
			outp (parameters.output_file.c_str())
{	
	unsigned int n_moments = material_data.get_n_moments();
	unsigned int n_groups = material_data.get_n_groups();
	
	spn_group.resize (n_groups);
	

  QGauss<1>  mu_quadrature(n_moments);  //get the value for \mu's according to the order of Sn approximation to be used.
	for (unsigned int m=0; m<n_moments; m++)		//m is the direction index
	{
	  mu.push_back(mu_quadrature.point(n_moments-1-m)(0)*2.0-1.0);
	  wt.push_back(mu_quadrature.weight(n_moments-1-m));
cout<<"angle index="<<m<<" : mu="<<mu[m]<<", wt="<<wt[m]<<endl;  //debug
	}

	Pnm.resize(n_moments);
	for(unsigned int order=0; order<n_moments; order++)
	{
		Pnm[order].resize(n_moments);
	  for(unsigned int m=0; m<n_moments; m++)
	    {
	      Pnm[order][m] = Legendre::Pn(order, mu[m]);
cout<<"Pnm("<<order<<","<<m<<") = "<<Pnm[order][m]<<endl;  //debug
	    }
	 }


	//material_data_old.correct_scat_moments();  //make optimal correction to the scattering moment (\sigma_s* = \sigma_s - (\sigma_{N/2} + \sigma_{N-1})/2 )
	
	for (unsigned int g=0; g<n_groups; g++)
		spn_group[g] = new SPN_group<dim> (g, material_data_new, material_data_old, coarse_grid, fe, Pnm);
		
	dsa = new DSA<dim> (material_data_new, parameters.boundary_conditions, parameters.boundary_value, coarse_grid, fe, mu, wt);  //initializing DSA object
	
	solution.resize(n_groups);	
	phi_even.reinit(n_groups, n_moments);
	phi_odd.reinit(n_groups, n_moments);
	phi_even_new = phi_even;
	phi_odd_new = phi_odd;
	phi_even_old = phi_even;
	phi_odd_old = phi_odd;
	phi_even_previous = phi_even;
	phi_odd_previous = phi_odd;
	q_minus.reinit(n_groups, n_moments/2);
	phi_old.resize(n_groups);
	J_old.resize(n_groups);
	response.resize(n_groups);
  
  dof_neighbour_dof.resize(n_groups);
  dof_repitition.resize(n_groups);
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
SPN<dim>::~SPN()
{
	// destructor of SPN (there is a problem here) 
	//delete common_dof_handler;
	delete dsa;
	for (unsigned int g=0; g<material_data.get_n_groups(); g++)
		delete spn_group[g];
		
		cout<<"you are here!!!"<<endl;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Triangulation<dim> SPN<dim>::initial_grid()
// this function reads the file for the assembly if it exists and we create the initial grid of our problem thanks to the informations that we have. It's here that we give the material properties of the cells
{
  Assert (false, ExcMessage ("1-D assembly data not available!"));
}

#if Dimension == 2

	template <>
	Triangulation<2> SPN<2>::initial_grid()
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
				// t.refine_global(2);  //debug,  isotropically refine mesh

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
Triangulation<3> SPN<3>::initial_grid()
{
  Triangulation<3> t;


  //std::ifstream in ("heatshield_medium_aligned.ucd");
  //std::ifstream in ("output.ucd"); 
  //std::ifstream in ("hs_align_extended20_cubit_mod.ucd");
  std::ifstream in ("out.inp");
  AssertThrow (in, ExcMessage ("Openning mesh file failed!"));
  
  
  //read-in the mesh from UCD file
  GridIn<3> grid_in;
  grid_in.attach_triangulation(t);
  grid_in.read_ucd(in);
  
  return t;
}
#endif  
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
Table<2, double> SPN<dim>::read_nodal_data()
{
  Table<2, double> nodal_data_reordered;  //reordered nodal_data field container
	
  std::string line;   //junk container
  Table<2, double> nodal_data;  //raw nodal data container
 
  unsigned int n_vertices = 0;
  unsigned int n_cells = 0;
  unsigned int num_ndata = 0;
  unsigned int n_comps = 0;
  
  std::ifstream in ("out.inp");
  AssertThrow (in, ExcMessage ("Openning mesh file failed!"));
  
  
  //Beginning of the UCD file
  char c;
  while( (c=in.get())=='#' )
    getline(in, line);
  in.putback(c);
  
  in >> n_vertices;
  in >> n_cells;
  getline(in, line);  //cut to the end of current line
  
  for(unsigned int i_row = 0; i_row < n_vertices + n_cells; i_row++)
    getline(in, line);
  
  //begin of nodal data section
  in >> n_comps;
  
  std::vector<unsigned int> comp_size(n_comps,1);
  for(unsigned i_comp=0; i_comp<n_comps; i_comp++)
  {
    in >> comp_size[i_comp];                  //read in the nodal data structure
    num_ndata += comp_size[i_comp];
  }
  
  nodal_data.reinit(n_vertices, num_ndata);   //resize the nodal_data table
  nodal_data_reordered.reinit(n_vertices, num_ndata);   //resize the nodal_data table
  
  getline(in, line);    //cut to the end of the nodal data structure line
  for(unsigned i_comp=0; i_comp<n_comps; i_comp++)
  {
    getline(in, line);    //skip the units specifications
//    std::cout<<line<<std::endl;  //debug
  }
  
  for(unsigned int i_row=0; i_row < n_vertices; i_row++)
  {
  	unsigned int i_vertex;
  	in >> i_vertex;
//  	std::cout<<i_vertex<<std::endl;  //debug
    for(unsigned int i_ndata=0; i_ndata<num_ndata; i_ndata++)
      in >> nodal_data(i_vertex-1, i_ndata);
  }
  
  //attach the mesh(Triangulation) to the dof_handler
  // so that the mesh can be viewed in VisIt
  DoFHandler<dim> dof_handler (coarse_grid);
  static const FE_Q<dim> finite_element(1);    //DOFs and vertices have 1-to-1 correspondence only for FE_Q(1)
  dof_handler.distribute_dofs (finite_element);
  
  
  //Auxiliary quadrature to map vertices to DOFS
  std::vector<unsigned int> vertex_to_dof(n_vertices);   //mapping from i_vertex to i_dof, globally	
  Quadrature<3> dummy_quadrature (finite_element.get_unit_support_points());  //dummy quadrature points to contain actually the support point on unit cell
	FEValues<3>   fe_values (finite_element, dummy_quadrature, update_quadrature_points);  //auxiliary FEValues object to map the support point from unit cell to real cell
  const unsigned int   dofs_per_cell = finite_element.dofs_per_cell;
  const unsigned int   vertices_per_cell = GeometryInfo<3>::vertices_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  AssertThrow (dofs_per_cell==vertices_per_cell, ExcMessage ("n_DOFs doesn't match n_vertices"));
  
  DoFHandler<3>::active_cell_iterator     
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
  	fe_values.reinit(cell);
  	cell->get_dof_indices(local_dof_indices);
  	
  	for(unsigned int i_vertex = 0; i_vertex < vertices_per_cell; i_vertex++)
  	{
  		vertex_to_dof[cell->vertex_index(i_vertex)] = local_dof_indices[i_vertex];  //cell vertice and dofs are indexed in the same order in deal.ii
  	}
  }
  
  //reorder the nodal_data according to DoF index
  for(unsigned int i_vertex = 0; i_vertex<n_vertices; i_vertex++)
    for(unsigned int i_ndata=0; i_ndata<num_ndata; i_ndata++)
  	{
  	  nodal_data_reordered(vertex_to_dof[i_vertex], i_ndata) = nodal_data(i_vertex, i_ndata);
  	}	
  
  // Output the vertex data extracted from UCD together with the vertex-to-dof mapping
  std::ofstream nodal_data_out("nodal_data.txt");
  AssertThrow (nodal_data_out, ExcMessage ("Creating Nodal Data output file failed!")); 
  for(unsigned int i_vertex=0; i_vertex < n_vertices; i_vertex++)
  {
    nodal_data_out << vertex_to_dof[i_vertex]<<" ";
    for(unsigned int i_ndata=0; i_ndata<num_ndata; i_ndata++)
    nodal_data_out << nodal_data(i_vertex, i_ndata) <<" ";  
    nodal_data_out << std::endl;
  }
  nodal_data_out.close(); 
  	
  	
  return nodal_data_reordered;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN<dim>::build_q_minus(unsigned int group, unsigned int m)
{ // this function is to project the known q-minus onto finite element basis consists of \grad B(r), 
	// so that the q_minus vector we get here can be directly fed into compute_phi_odd function
	 
	const RHS_psi_minus<dim> right_hand_side(mu,wt);
	QGauss<1>  quadrature_formula(fe.degree+1);		//assume one degree higher than FE will be enough for line integral below
	const unsigned int   dofs_per_cell = fe.dofs_per_cell; 
	const unsigned int   n_q_points    = quadrature_formula.size(); 
	
	Quadrature<dim> dummy_quadrature (fe.get_unit_support_points());  //dummy quadrature points to contain actually the support point on unit cell
	FEValues<dim>   fe_values (fe, dummy_quadrature, update_quadrature_points);  //auxiliary FEValues object to map the support point from unit cell to real cell
		
	typename DoFHandler<dim>::active_cell_iterator	
		cell = spn_group[group]->dof_handler.begin_active(),
		endc = spn_group[group]->dof_handler.end();
	
	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
		
		fe_values.reinit (cell);
		Point<dim> vertex_0;
    vertex_0 = cell->vertex(0);
 
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			// Get DOF Support Point
			const Point<dim>& support_point = fe_values.quadrature_point (i);  //mapped support point on real "cell"
 			
 			//We need to integrate the external source from a reference point (chosen to be the first vertex within current cell).
 			//This is because we are using gradient basis function. Thus the dof values should be integrated values.
 			for(unsigned int d = 0; d<dim; d++)
 			{
 				double width = support_point(d) - vertex_0(d); //the width that we will be integrating over along an axis (x/y/z) direction
 			  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
 			  {
 				  double xq = quadrature_formula.point(q_point)(0)*width + vertex_0(d);		//xq = quadrature points along x axis
	  		  double wt = quadrature_formula.weight(q_point);
	  		  Point<dim> scaled_q_point;
	  		  for(unsigned int i_d=0; i_d<d; i_d++)
	  		  	scaled_q_point(i_d) = support_point(i_d);
	  		  scaled_q_point(d) = xq;
	  		  for(unsigned int i_d= d+1; i_d<dim; i_d++)
	  		    scaled_q_point(i_d) = vertex_0(i_d);
	  		
	  	  	q_minus[group][m](i_cell*dofs_per_cell+i) += right_hand_side.get_vector_source(scaled_q_point, group, domain_size, m)[d]*wt*width;  //Assume single energy group
 			  }
 		  }
 		}
	
//	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
//  {
//		
//		fe_values.reinit (cell);
//		Point<dim> vertex_0 (cell->vertex(0)(0),cell->vertex(0)(1));  //get the left-bottom vertex of the cell as the reference point in this cell
//		for (unsigned int i=0; i<dofs_per_cell; ++i)
//		{
//			// Get DOF Support Point
//			const Point<dim>& support_point = fe_values.quadrature_point (i);  //mapped support point on real "cell"
// 			
// 			//We need to integrate the external source from a reference point (chosen to be the first vertex within current cell).
// 			//This is because we are using gradient basis function. Thus the dof values should be integrated values.
// 			// Here we are doing 2-D line integral (for 2-D case, can be generalized to 3D easily);
//			//integrate along x axis
//			double width = support_point(0) - vertex_0(0); //the width that we will be integrating over along x direction
// 			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
// 			{
// 				double xq = quadrature_formula.point(q_point)(0)*width + vertex_0(0);		//xq = quadrature points along x axis
//	  		double wt = quadrature_formula.weight(q_point);
//	  		Point<dim> scaled_q_point(xq, vertex_0(1));		//integrate along y=y(vertex_0);
//	  		
//	  		q_minus[group][m](i_cell*dofs_per_cell+i) += right_hand_side.get_vector_source(scaled_q_point, group, domain_size, m)[0]*wt*width;  //Assume single energy group
// 			}
// 			//integrate along y axis
//			double height = support_point(1) - vertex_0(1); //the heigt that we will be integrating over along y direction
// 			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
// 			{
// 				double yq = quadrature_formula.point(q_point)(0)*height + vertex_0(1);  //yq = quadrature points along y axis
//	  		double wt = quadrature_formula.weight(q_point);
//	  		Point<dim> scaled_q_point(support_point(0), yq);	//integrate along x=x(current_support_point);
//	  		
//	  		q_minus[group][m](i_cell*dofs_per_cell+i) += right_hand_side.get_vector_source(scaled_q_point, group, domain_size, m)[1]*wt*height;  //Assume single energy group
// 			}
// 			
// 		}
  }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN<dim>::assemble_system (unsigned int m, unsigned int group, bool suppress_source)
{	// this function is used to make the the system of equations for the direct problem

  // initialize and purge some containers
//  spn_group[group]->solution_moment[moment].reinit (spn_group[group]->dof_handler.n_dofs()); //we create the pattern of the solution
//	spn_group[group]->system_rhs.reinit (spn_group[group]->dof_handler.n_dofs()); //we create the pattern of the right-hand-side
//	spn_group[group]->system_matrix.reinit (spn_group[group]->sparsity_pattern); // now the matrix receive her pattern
  
	RHS<dim> right_hand_side;
	right_hand_side.set_suppress(suppress_source);
	bool diffusion = false;
	if(material_data.get_n_moments() == 1)		//for isotropic case, we are solving one diffusion equation
		diffusion = true;
	
	QGauss<dim>  quadrature_formula(2*fe.degree);   //assuming linear material property
  FEValues<dim> fe_values (fe, quadrature_formula, 
								update_values | update_gradients | update_q_points | update_JxW_values);	
	
	const QGauss<dim-1> face_quadrature_formula(2*fe.degree);		
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
		cell = spn_group[group]->dof_handler.begin_active(),
		endc = spn_group[group]->dof_handler.end();
		
	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
    {
			fe_values.reinit (cell);
	 
			cell_matrix = 0;
			cell_rhs = 0;

			double st = material_data_new.get_total_XS(cell->material_id(), group);  //get the total cross-section
			double ss0 = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0th-moment of the scattering cross-section
			double st_origin = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
			double sa = st_origin - ss0;  //used for manufacturered solution
			std::vector<double> st_t(n_q_points);
			std::vector<double> sa_t(n_q_points);
			std::vector<double> T4(n_q_points);
			std::vector<double> diffusion_coefficient(n_q_points);
			double st_t_zero = k_abs_min;    //1e-5 m.f.p.
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
			{
			  st_t[q_point] = material_data_new.get_total_XS_t(i_cell, q_point, group);
		    sa_t[q_point] = material_data.get_total_XS_t(i_cell, q_point, group) - ss0;
		    if(!diffusion)
		      diffusion_coefficient[q_point] = mu[m]/std::max(st_t[q_point], st_t_zero);   // we create the diffusion coefficient
			  else
				  diffusion_coefficient[q_point] = 1.0/(3.0*std::max(st_t[q_point], st_t_zero));   // we create the diffusion coefficient
		    T4[q_point] = material_data.get_T4(i_cell, q_point);
		  }
		   
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					for (unsigned int j=0; j<dofs_per_cell; ++j)		//we give the values to the matrix
					{
						if(!diffusion)	
						{	
							cell_matrix(i,j) += (mu[m]* diffusion_coefficient[q_point] *
											fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) +
											st_t[q_point] *									
											fe_values.shape_value (i,q_point) * fe_values.shape_value (j,q_point)) *
											fe_values.JxW (q_point);


							// this is where we take care of the div(Q-) term
							// contribution to RHS due to odd parity flux
				  		for (unsigned int kj=1; kj < material_data_new.get_n_moments(); kj = kj+2)
				  			cell_rhs(i) += (diffusion_coefficient[q_point]*
				  							double(2*kj+1)*
				  							material_data_old.get_moment_XS(cell->material_id(), group, kj)*
				  							Pnm[kj][m] *
				  		  				phi_odd_old[group][kj](i_cell*dofs_per_cell + j) *
				  							fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) * 
				  							fe_values.JxW (q_point) );
				  
				  		// contribution to RHS due to odd parity external source					
				  		if(!suppress_source)
				  			cell_rhs(i) += (mu[m]* diffusion_coefficient[q_point]*
			  										q_minus[group][m](i_cell*dofs_per_cell + j)*
			  									 fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) * 
			  									 fe_values.JxW (q_point) );
			  		}
			  		else	//if the scattering is anisotropic
			  			cell_matrix(i,j) += (diffusion_coefficient[q_point] *
											fe_values.shape_grad (i, q_point) * fe_values.shape_grad (j, q_point) +
											sa_t[q_point] *									
											fe_values.shape_value (i,q_point) * fe_values.shape_value (j,q_point)) *
											fe_values.JxW (q_point);
			    }
			
					// we give the values for the right-hand-side due to even parity external source
					cell_rhs(i) +=( 
									 sa_t[q_point]*right_hand_side.get_source (fe_values.quadrature_point (q_point), group, T4[q_point], diffusion) *
								   fe_values.shape_value (i, q_point) *
								   fe_values.JxW (q_point));
								   
				}	  

	
			//we put the Robin boundary conditions if they exist
			if (boundary_exist != parameters.boundary_conditions.end()) 
			{
				RC_exist=true;
				for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) 		// we make a loop over all the face, if we have a face where we have a boundary 
																								// condition we go to the if
				{
					unsigned int side = cell->face(face)->boundary_indicator()-1;
					if (cell->at_boundary(face) && (parameters.boundary_conditions[side] == 2))
					{
						fe_face_values.reinit (cell, face);
						
						std::vector<double> T4_face(n_face_q_points);
            fe_face_values.get_function_values(material_data.get_T4_vertex(), T4_face);
									
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)	// we modify the matrix and the right-hand-side because of the boundary condition
						{
							for (unsigned int i=0; i<dofs_per_cell; ++i)
							{	
								if(!diffusion)
								{
	  							cell_rhs(i) +=  (mu[m] *
	  											right_hand_side.get_Jinc (fe_face_values.quadrature_point (q_point),
	  											(side==2?4.0*sigma_Boltzmann*T4_face[q_point]:parameters.boundary_value[side]), group, m+1, mu, wt)  *
	  											fe_face_values.shape_value(i,q_point) *
	  											fe_face_values.JxW(q_point));
	  							
	  
	  							for (unsigned int j=0; j<dofs_per_cell; j++)	 
	  								cell_matrix(i,j) +=  (mu[m] *
	  													fe_face_values.shape_value(i,q_point) *
	  													fe_face_values.shape_value(j,q_point)*
	  													fe_face_values.JxW(q_point));
	  						}
	  						else		//boundary condition for diffusion equation
	  						{
	  							cell_rhs(i) +=  ( 0.5*
	  											right_hand_side.get_Jinc (fe_face_values.quadrature_point (q_point),
	  											(side==2?4.0*sigma_Boltzmann*T4[q_point]:parameters.boundary_value[side]), group, m+1, mu, wt)  *
	  											fe_face_values.shape_value(i,q_point) *
	  											fe_face_values.JxW(q_point));
	  							
	  
	  							for (unsigned int j=0; j<dofs_per_cell; j++)	 
	  								cell_matrix(i,j) +=  ( 0.5*
	  													fe_face_values.shape_value(i,q_point) *
	  													fe_face_values.shape_value(j,q_point)*
	  													fe_face_values.JxW(q_point));
	  						}
							}
						}
					}
				}
			}
	 
			cell->get_dof_indices (local_dof_indices);
	
			for (unsigned int i=0; i<dofs_per_cell; ++i)	// we put the matrix and the right-hand-side of the cell into the matirx and the right-hand-side of the system
			{	
				for (unsigned int j=0; j<dofs_per_cell; ++j)
					spn_group[group]->system_matrix.add (local_dof_indices[i],
										local_dof_indices[j],
										cell_matrix(i,j));
	
				spn_group[group]->system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}	
    }
	if(!diffusion)
	//we take care about the term in the right-hand-side due to even parity scattering
		for (unsigned int kj=0; kj < material_data_new.get_n_moments(); kj = kj+2)
		{ 	
			//!!!!!!!!!Need Revision!!!!!!!!!!!!!!//
			{ 	
				spn_group[group]->assemble_scattering_matrix (kj, m, 2*kj+1); 
				spn_group[group]->scattering_matrix[kj].vmult_add (spn_group[group]->system_rhs, phi_even_old[group][kj]);
			}
		}

	// we take care about the hanging nodes
	spn_group[group]->hanging_node_constraints.condense (spn_group[group]->system_matrix);		
	spn_group[group]->hanging_node_constraints.condense (spn_group[group]->system_rhs);
  
	std::map<unsigned int,double> boundary_values;	//  we use that if we some Dirichlet conditions, not implemented yet
	for (unsigned int i=0; i<parameters.boundary_conditions.size();i++)
		if (parameters.boundary_conditions[i]==1)
			VectorTools::interpolate_boundary_values (spn_group[group]->dof_handler,
                                            i,
                                            BoundaryValues<dim>(),
                                            boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
                                      spn_group[group]->system_matrix,
                                      spn_group[group]->solution_moment[m],
                                      spn_group[group]->system_rhs);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::compute_phi_even(unsigned int group, unsigned int n_moments)
	//compute even moment flux after solving for angular flux
{
	Vector<double> temp;

	  for(unsigned int moment=0; moment<n_moments; moment=moment+2)
	  {
		  phi_even_new[group][moment].reinit(spn_group[group]->n_dofs,false);
		  for(int m=0; m<double(n_moments/2.0); m++)
		  {
			  temp = spn_group[group]->solution_moment[m];

			  phi_even_new[group][moment] += temp*=(2.0*Pnm[moment][m]*wt[m]);
			  if(n_moments==1)
				  phi_even_new[group][moment] = spn_group[group]->solution_moment[m];
		  }
		  cout<<"L2[phi("<<moment<<")]="<<phi_even_new[group][moment].l2_norm()<<endl;  //debug
	  }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::compute_phi_odd(unsigned int group, unsigned int n_moments, bool suppress_source)
	//compute odd moment flux after solving for angular flux. Store on DG dofs;
{
	double temp;
	for(unsigned int moment=1; moment<n_moments; moment=moment+2)
	{		
		std::vector<double> st_t(spn_group[group]->n_dofs);
		double st_t_zero = k_abs_min;    //1e-3 m.f.p.
		for(unsigned int i_dof = 0; i_dof < spn_group[group]->n_dofs; i_dof++)
			st_t[i_dof] = std::max(material_data_new.get_k_abs_vertex()(i_dof), st_t_zero);
     
		
		const unsigned int   dofs_per_cell = spn_group[group]->fe.dofs_per_cell;
		std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
		typename DoFHandler<dim>::active_cell_iterator	
		cell = spn_group[group]->dof_handler.begin_active(),
		endc = spn_group[group]->dof_handler.end();
		
		for (unsigned int i=0; cell!=endc; ++cell, ++i) 
		{
			double st = material_data_new.get_total_XS(cell->material_id(), group);  //get the total cross-section  
			double ss = material_data_old.get_moment_XS(cell->material_id(), group, moment);  //get the scattering moment cross-section
		  
			cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
			
			for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
			{
				temp = 0.0;  //reset temporary container for phi_odd[moment][n], where n is the discontinuous basis function index
				for(unsigned int m=0; m<n_moments/2; m++)
				{
					temp += -mu[m]/st_t[local_dof_indices[j]]*spn_group[group]->solution_moment[m](local_dof_indices[j])*2.0*Pnm[moment][m]*wt[m];  //grad(psi^+) term in odd-parity equations
					if(!suppress_source)
						temp += 1.0/st_t[local_dof_indices[j]]*q_minus[group][m](i*dofs_per_cell+j)*2.0*Pnm[moment][m]*wt[m];  //odd parity external source q^- term, due to Legendre orthogonality
//				  cout<<st_t[local_dof_indices[j]]<<endl;   //debug
				}
				temp += ss/st_t[local_dof_indices[j]]*phi_odd_old[group][moment](i*dofs_per_cell+j); //only the "moment"-th moment remains, due to Legendre orthogonality
				phi_odd_new[group][moment](i*dofs_per_cell+j) = temp;  //temp is saved as phi_odd_new[moment] value at position (i*dofs_per_cell+j)
			}
		}
		cout<<"L2[phi("<<moment<<")]="<<phi_odd_new[group][moment].l2_norm()<<endl;  //debug
	}
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::attach_neighbour_dofs(unsigned int idof, unsigned int neighbour_idof, unsigned int group)
	{
		bool neighbour_exists = false;
		for(unsigned int k=0;k<dof_neighbour_dof[group][idof].size();k++)
		  if(dof_neighbour_dof[group][idof][k] == neighbour_idof)
			  neighbour_exists = true;
						
	  if(neighbour_exists == false)
			dof_neighbour_dof[group][idof].push_back(neighbour_idof);
	}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::build_dof_repitition_neigbourhood(unsigned int group)
{

	dof_repitition[group].reinit(spn_group[group]->n_dofs,false);  //reset the shared DOF counter;
	dof_neighbour_dof[group].resize(spn_group[group]->n_dofs);     //give the size of first dimension of the dof_sharing_cells = n_dofs
		
	const unsigned int   dofs_per_cell = spn_group[group]->fe.dofs_per_cell;
	std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
	typename DoFHandler<dim>::active_cell_iterator	
	cell = spn_group[group]->dof_handler.begin_active(),
	endc = spn_group[group]->dof_handler.end();
	
	for (unsigned int i=0; cell!=endc; ++cell, ++i) 
	{
		cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
		if(dim==2 && fe.degree==1)
		{
			for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
  		{
  			dof_repitition[group](local_dof_indices[j])++;  //count the repitition of current DOF, i.e., how many cells share this DOF
  			if(j==0)
    		{
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
    		}
    		else if(j==1)
    		{
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
    		}
    		else if(j==2)
    		{
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
    		}
    		else if(j==3)
    		{
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
    			attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
    		}
    	}
	  }
		else if(dim==3 && fe.degree == 1)
		{
			for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
  		{
  		  dof_repitition[group](local_dof_indices[j])++;  //count the repitition of current DOF, i.e., how many cells share this DOF
  		  if(j==0)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  			}
  			else if(j==1)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  			}
  			else if(j==2)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  			}
  			else if(j==3)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  			}
  			else if(j==4)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[6], group);
  			}
  			else if(j==5)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  			}
  			else if(j==6)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  			}
  			else if(j==7)
  			{
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  				attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[6], group);
  			}
  		}
  	}
		else if(dim==2 && fe.degree == 2)
		{
  		for (unsigned int j=0; j<dofs_per_cell; ++j)  //j = local dof index
  		{
  				dof_repitition[group](local_dof_indices[j])++;  //count the repitition of current DOF, i.e., how many cells share this DOF
  				if(j==0)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[6], group);
  				}
  				else if(j==1)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[6], group);
  				}
  				else if(j==2)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  				}
  				else if(j==3)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  				}
  				else if(j==4)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[8], group);
  				}
  				else if(j==5)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[8], group);
  				}
  				else if(j==6)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[0], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[1], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[8], group);
  				}
  				else if(j==7)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[2], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[3], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[8], group);
  				}
  				else if(j==8)
  				{
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[4], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[5], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[6], group);
  					attach_neighbour_dofs(local_dof_indices[j], local_dof_indices[7], group);
  				}
  		}
  	}
	}
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::compute_angular_quadrature(unsigned int n_moments, 
																					std::vector<std::vector<double> > &Pnm,
																					std::vector<double> &mu,
																					std::vector<double> &wt)
//angular quadrature need to be re-compute for each level of AMG calculation
{
		mu.resize(n_moments);
		wt.resize(n_moments);
	  QGauss<1>  mu_quadrature(n_moments);  //get the value for \mu's according to the order of Sn expansion to be used.
		for (unsigned int m=0; m<n_moments; m++)
		{
		  mu[m] = mu_quadrature.point(n_moments-1-m)(0)*2.0-1.0;
		  wt[m] = mu_quadrature.weight(n_moments-1-m);
	cout<<"angle index="<<m<<" : mu="<<mu[m]<<", wt="<<wt[m]<<endl;  //debug
		}
	
		Pnm.resize(n_moments);
		for(unsigned int order=0; order<n_moments; order++)
		{
			Pnm[order].resize(n_moments);
		  for(unsigned int angle=0; angle<n_moments; angle++)
		    {
		      Pnm[order][angle] = Legendre::Pn(order, mu[angle]);
	cout<<"Pnm("<<order<<","<<angle<<") = "<<Pnm[order][angle]<<endl;  //debug
		    }
		 }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::reinit_phi_even_odd(unsigned int group)
{

	  for(unsigned int moment=0; moment<material_data.get_n_moments(); moment=moment+2)
		  phi_even[group][moment].reinit(spn_group[group]->n_dofs,false);
		
	  for(unsigned int moment=1; moment<material_data.get_n_moments(); moment=moment+2)   //give initial value for phi_odd
		  phi_odd[group][moment].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false);	//number of DG dofs is different from CG dofs

}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::compute_Leakage(unsigned int group)
{
	double source = 0.0;  //total particle gain
	double sink = 0.0;    //total particle loss
	
	compute_angular_quadrature(material_data.get_n_moments(), Pnm, mu, wt);
	
	bool suppress_source = false;
	RHS<dim> right_hand_side;
	right_hand_side.set_suppress(suppress_source);
	bool diffusion = false;
	if(material_data.get_n_moments() == 1)		//for isotropic case, we are solving one diffusion equation
		diffusion = true;
		
  //== Compute the renormalization factor \alpha
	 double alpha = 0;    //renormalization factor
 	 double half_range_integral = 1.0/2*0.5;        //actual half range current, auxiliary variable to compute \alpha
 	 double half_range_sum = 0;   //quadrature half range current, auxiliary variable to compute \alpha
 	 for(unsigned i=0; i<wt.size()/2; i++)
 	   half_range_sum += 1.0/2*mu[i]*wt[i]*2;
 	   
 	 if(half_range_sum!=0.0)
 	   alpha = half_range_integral/half_range_sum;
 	 else
 	 	 alpha = 1.0;
	
	const QGauss<dim-1> face_quadrature_formula(fe.degree +1);		
	const unsigned int n_face_q_points = face_quadrature_formula.size();
	FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
								update_values | update_q_points | update_normal_vectors | update_JxW_values);
 

	std::vector<int>::iterator boundary_exist;					//we check if we have some Robin boundary condtions
	boundary_exist=std::find(parameters.adjoint_boundary_conditions.begin(), parameters.adjoint_boundary_conditions.end(), 2);
	bool RC_exist=false;
	
	
	typename DoFHandler<dim>::active_cell_iterator	
		cell = spn_group[group]->dof_handler.begin_active(),
		endc = spn_group[group]->dof_handler.end();
		
	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {

		
		if (boundary_exist != parameters.adjoint_boundary_conditions.end()) 
		{
			RC_exist=true;
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) 		// we make a loop over all the face, if we have a face where we have a boundary 
																												// condition we go to the if
			{
				unsigned int side = cell->face(face)->boundary_indicator()-1;
  			if ( cell->at_boundary(face) && (parameters.adjoint_boundary_conditions[side] == 2)
        	   && (parameters.adjoint_boundary_value[side] == 1) )
  			{
  				fe_face_values.reinit (cell, face);
  				
  				std::vector<double> T4_face(n_face_q_points);
          fe_face_values.get_function_values(material_data.get_T4_vertex(), T4_face);
  				
  				for(int m=0; m<double(material_data.get_n_moments()/2.0); m++)
  	      {
  	       	Vector<double> psi_plus_m = spn_group[group]->solution_moment[m];
  				
  				  std::vector<double> face_psi_values(n_face_q_points);
            fe_face_values.get_function_values(psi_plus_m,face_psi_values);	
            
  									
  				  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)	// we modify the matrix and the right-hand-side because of the boundary condition
  				  {
  					
   
       		        
              sink +=  std::fabs( mu[m] ) *
                       alpha*(2.0*face_psi_values[q_point] ) *  // \psi(\Omega) = 2*psi+*(\Omega) - f(-\Omega), \Omega*n >0
                       wt[m]*                       //integrate over all directions                                                                                                  
                       fe_face_values.JxW(q_point);        //integrate over cell volume
                           
                           
              source += std::fabs( mu[m] ) *
                       (  - right_hand_side.get_Jinc (fe_face_values.quadrature_point (q_point),
  	  											(side==2?4.0*sigma_Boltzmann*T4_face[q_point]:parameters.boundary_value[side]), group, m+1, mu, wt) ) *  // \psi(\Omega) = 2*psi+*(\Omega) - f(-\Omega), \Omega*n >0
                       wt[m]*                       //integrate over all directions                                                                                                  
                       fe_face_values.JxW(q_point);        //integrate over cell volume
            }
          }
        }
      }
    }
		
	}
	
	response[group] = sink;
  cout<<"Leakage of Interest: "<<sink<<endl;
  cout<<"Source  of Interest: "<<source<<endl;
  cout<<"Half-Range-Current : "<<response[group]<<endl;
	
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<int dim>
void SPN<dim>::check_conservation(unsigned int group)
{
	double source = 0.0;  //total particle gain
	double sink = 0.0;    //total particle loss
	double conservation = 0.0;  //conservation = (source-sink)/source
	
	compute_angular_quadrature(material_data.get_n_moments(), Pnm, mu, wt);
	
	bool suppress_source = false;
	RHS<dim> right_hand_side;
	right_hand_side.set_suppress(suppress_source);
	bool diffusion = false;
	if(material_data.get_n_moments() == 1)		//for isotropic case, we are solving one diffusion equation
		diffusion = true;
	
	QGauss<dim>  quadrature_formula(2*fe.degree); 
  FEValues<dim> fe_values (fe, quadrature_formula, 
								update_values | update_gradients | update_q_points | update_JxW_values);	
	
	const QGauss<dim-1> face_quadrature_formula(2*fe.degree );		
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
		cell = spn_group[group]->dof_handler.begin_active(),
		endc = spn_group[group]->dof_handler.end();
		
	for (unsigned int i_cell=0; cell!=endc; ++cell, ++i_cell) 
  {
	  fe_values.reinit (cell);
	  
		double ss0 = material_data.get_moment_XS(cell->material_id(), group, 0); //get 0th-moment of the scattering cross-section
		double st_origin = material_data.get_total_XS(cell->material_id(), group);  //get the total cross-section
		double sa = st_origin - ss0;  //used for manufacturered solution
		std::vector<double> st_t(n_q_points);
		std::vector<double> sa_t(n_q_points);
		std::vector<double> T4(n_q_points);
		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		{
			st_t[q_point] = material_data_new.get_total_XS_t(i_cell, q_point, group);
		  sa_t[q_point] = material_data.get_total_XS_t(i_cell, q_point, group) - ss0;
		  T4[q_point] = material_data.get_T4(i_cell, q_point);
		}
		
		std::vector<double> phi_values(quadrature_formula.size());
		fe_values.get_function_values(phi_even[group][0],phi_values);	
		

		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		{
			sink += phi_values[q_point]
			        *sa_t[q_point]
			        *fe_values.JxW(q_point);		    //we calculate the absorption rate
			
			
			for(int m=0; m<double(material_data.get_n_moments()/2.0); m++)
		  {
		  	source += sa_t[q_point]*right_hand_side.get_source (fe_values.quadrature_point (q_point), group, T4[q_point], diffusion) *
		  	          2.0 * wt[m]*
								  fe_values.JxW (q_point);    //we calculate the volumetric source rate
		  }	  
		}
		
		if (boundary_exist != parameters.boundary_conditions.end()) 
		{
			RC_exist=true;
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) 		// we make a loop over all the face, if we have a face where we have a boundary 
																												// condition we go to the if
		  {
  		  unsigned int side = cell->face(face)->boundary_indicator()-1;
  			if (cell->at_boundary(face) && (parameters.boundary_conditions[side] == 2))
  			{
  				fe_face_values.reinit (cell, face);
  									
  				for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)	// we modify the matrix and the right-hand-side because of the boundary condition
  				{
  					for(int m=0; m<double(material_data.get_n_moments()/2.0); m++)
  	        {
  	         	Vector<double> psi_plus_m = spn_group[group]->solution_moment[m];
   
              std::vector<double> face_psi_values(n_face_q_points);
              fe_face_values.get_function_values(psi_plus_m,face_psi_values);	
              std::vector<double> T4_face(n_face_q_points);
              fe_face_values.get_function_values(material_data.get_T4_vertex(), T4_face);
       		        
              sink +=  std::fabs( mu[m] ) *
                       (2.0*face_psi_values[q_point]
                        - right_hand_side.get_Jinc (fe_face_values.quadrature_point (q_point),
  	  											(side==2?4.0*sigma_Boltzmann*T4_face[q_point]:parameters.boundary_value[side]), group, m+1, mu, wt) ) *  // \psi(\Omega) = 2*psi+*(\Omega) - f(-\Omega), \Omega*n >0
                       wt[m]*                       //integrate over all directions                                                                                                  
                       fe_face_values.JxW(q_point);        //integrate over cell volume
                           
                           
              source += std::fabs( mu[m] ) *
                        right_hand_side.get_Jinc (fe_face_values.quadrature_point (q_point),
  	  											(side==2?4.0*sigma_Boltzmann*T4_face[q_point]:parameters.boundary_value[side]), group, m+1, mu, wt)  *    // \psi(\Omega) = f(\Omega), \Omega*n <0
                        wt[m]*                       //integrate over all directions                                                                                                  
                        fe_face_values.JxW(q_point);        //integrate over cell volume
            }
          }
        }
      }
    }
		
	}
	
	conservation = (source-sink)/source;
  cout<<"Particle Conservation : "<<conservation<<endl;
	
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>  
void SPN<dim>::output (int cycle) const
{	// this function makes the output for the scalar flux of the direct and the adjoint problem
	for(unsigned int group = 0; group<material_data.get_n_groups(); group++)
  {
  	DataOut<dim> data_out;
  	data_out.attach_dof_handler (spn_group[group]->dof_handler);
	  data_out.add_data_vector (phi_even[group][0], "solution");
	  Vector<double> T_vertex(material_data.get_T_vertex());
	  Vector<double> k_abs_vertex(material_data.get_k_abs_vertex());
	  data_out.add_data_vector (T_vertex, "temperature");
	  data_out.add_data_vector (k_abs_vertex, "k_absorption");
	  data_out.add_data_vector (dof_repitition[group], "dof_repitition");
	  
	  data_out.build_patches ();
 
	  std::ostringstream filename;
	
	  filename << "solution-"<< group << ".vtk";
 
  	std::ofstream output (filename.str().c_str());
	  data_out.write_vtk (output);	
	  
	  
	  //===== Output the Response  ========
    std::ostringstream response_filename;
    response_filename<<"Response_SPn-"<<group<<".txt";
    std::ofstream Response_out(response_filename.str().c_str());
    Response_out.precision(12);
    Response_out<<response[group]<<std::endl;
    Response_out.close();
	  
	}
	
	
	
	//===== Store the phi_even and phi_odd to the output file, for reconstruction in Sn code
	for(unsigned int group = 0; group<material_data.get_n_groups(); group++)
  {
  	std::ostringstream output_filename;
  	output_filename<<parameters.spn_output_file<<"-"<<group;
  	std::ofstream f_out(output_filename.str().c_str());    
  	f_out.precision(12);
  	AssertThrow (f_out, ExcMessage ("Creating output file failed!"));
  	unsigned int ndof_even = spn_group[group]->n_dofs;
  	unsigned int ndof_odd = spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell;
  	unsigned int n_moments = material_data.get_n_moments();
  	
  	f_out<<"#Even Moments "<<ndof_even<<" "<<n_moments<<endl;
  	for(unsigned int idof=0; idof<ndof_even; idof++)
  	{
  		for(unsigned int moment=0; moment<n_moments; moment=moment+2)
  			f_out<<phi_even[group][moment](idof)<<" ";
  		f_out<<"E ";
  		
  		f_out<<dof_repitition[group](idof)<<" ";
  		
  		for(unsigned int i_nb=0; i_nb<dof_neighbour_dof[group][idof].size(); i_nb++)
  		  f_out<<dof_neighbour_dof[group][idof][i_nb]<<" ";
  		
  		f_out<<"#";
  		f_out<<endl;
  	}
  	f_out<<"#Odd Moments "<<ndof_odd<<" "<<n_moments<<endl;
  	for(unsigned int idof=0; idof<ndof_odd; idof++)
  	{
  		for(unsigned int moment=1; moment<n_moments; moment=moment+2)
  			f_out<<phi_odd[group][moment](idof)<<" ";
  		f_out<<endl;
  	}
  	f_out<<"#EvenParity Angular Flux "<<ndof_even<<" "<<n_moments<<endl;
  	for(unsigned int idof=0; idof<ndof_even; idof++)
  	{
  		for(unsigned int m=0; m<n_moments/2; m++)
  			f_out<<spn_group[group]->solution_moment[m](idof)<<" ";
  		f_out<<endl;
  	}
  	f_out.close();
  	
  			//output J at cell center;
  			std::ostringstream J_vector_filename;
  			J_vector_filename<<"J_vector-"<<group<<".txt";
  			std::ofstream grad_out(J_vector_filename.str().c_str());
  				
  			std::vector<std::vector<double> > grad_vector(spn_group[group]->triangulation.n_active_cells());
  			for(unsigned int i=0; i<grad_vector.size(); i++)
  	 			grad_vector[i].resize(2*dim, 0.0);
  			Quadrature<dim> dummy_quadrature (1);
  			FEValues<dim> fe_values (fe, dummy_quadrature, update_q_points | update_gradients  | update_JxW_values);
  			
  			const unsigned int   dofs_per_cell = spn_group[group]->fe.dofs_per_cell;
  			std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
  				
  			typename DoFHandler<dim>::active_cell_iterator	
  			cell = spn_group[group]->dof_handler.begin_active(),
  			endc = spn_group[group]->dof_handler.end();
  			
  			for (unsigned int i=0; cell!=endc; ++cell, ++i) 
  			{
  				fe_values.reinit(cell);
  				cell->get_dof_indices (local_dof_indices);  //get mapping from local to global dof
  				for (unsigned int j=0; j<dofs_per_cell; ++j)
        	{
            for(unsigned int component=0; component<dim; component++)
            {
              grad_vector[i][component+dim] += fe_values.shape_grad (j, 0)[component] * phi_odd[group][1](i*dofs_per_cell+j);
            }
  //          grad_vector[i][2] += fe_values.shape_grad (j, 0)[0] * q_minus[0](i*dofs_per_cell+j)/mu[0];
  //          grad_vector[i][3] += fe_values.shape_grad (j, 0)[1] * q_minus[0](i*dofs_per_cell+j)/mu[0];
  //          grad_vector[i][2] += fe_values.shape_grad (j, 0)[0] * phi_even[0](local_dof_indices[j]);
  //          grad_vector[i][3] += fe_values.shape_grad (j, 0)[1] * phi_even[0](local_dof_indices[j]);
            for(unsigned int component=0; component<dim; component++)
            {
              grad_vector[i][component] = fe_values.quadrature_point (0)[component];
            }
        	}
        }
        for(unsigned int i=0; i<grad_vector.size(); i++)
        {
          for(unsigned int component=0; component<2*dim; component++)
  				  grad_out<<grad_vector[i][component]<<" ";
  				grad_out<<endl;
  			}
  			grad_out.close();
	}
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void SPN<dim>::run () 
 {	// this function calls all the interesting function. Here we make all the loops over the number over the moments for the diret problem and the adjoint
	std::cout.precision(12);
	int n_refinement=parameters.n_refinement_cycles;
	double conv_tol = parameters.conv_tol;
	unsigned int dofs = 0;
	bool all_output = parameters.all_output;

	for(int cycle=0; cycle<=n_refinement; cycle++) // we make this loop for each refinement 
	{	
		Timer timer;
		timer.start ();
		
		std::cout<<"Cycle = "<<cycle<<std::endl;
		
		for(unsigned int group = 0; group<material_data.get_n_groups(); group++)   //loop through all groups
		{
  		build_dof_repitition_neigbourhood(group);
  			  		
  		if (cycle!=0)		// here we refine the meshes except if it's the first cycle because we don't have a solution yet.
  			;//refine_grid(cycle); //we are not refining the mesh now;
  		else
  	  {
  			for(unsigned int m=0; m<material_data.get_n_moments()/2; m++)
  			{
  				q_minus[group][m].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false); 
  				build_q_minus(group,m);
  			}
//  			compute_phi_even(group, material_data.get_n_moments());  //give initial value for phi_even_new
  			
  			for(unsigned int moment=0; moment<material_data.get_n_moments(); moment=moment+2)   
  			{
  				phi_even_old[group][moment].reinit(spn_group[group]->n_dofs,false);  
  				phi_even_new[group][moment].reinit(spn_group[group]->n_dofs,false);
  				phi_even_previous[group][moment].reinit(spn_group[group]->n_dofs,false);
  			}
  			for(unsigned int moment=1; moment<material_data.get_n_moments(); moment=moment+2) 
  			{
  				phi_odd_old[group][moment].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false);
  				phi_odd_new[group][moment].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false);
  				phi_odd_previous[group][moment].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false);
  	    }
//  			compute_phi_odd(group, material_data.get_n_moments(),false);  //give initial value for phi_odd_new
  			phi_old[group] = phi_even_new[group][0];  //initial guess for phi_old, for P1_SA;
  			J_old[group] = phi_odd_new[group][1];  //initial guess for J_old, for P1_SA
  		}
  	}
  			
  	std::vector<double> conv(material_data.get_n_groups(), 10);
  	bool not_converged = true;
  	for(unsigned int j=0; not_converged && j<=20 ;j++) // we solve the direct problem. we need to iterate over all the moments until we converge
  	{
  		not_converged = false;  //reset the not_conerged flag to FALSE for all groups
  		cout<<"Begin Source Iteration, j = "<<j<<endl;
  		for(unsigned int group = 0; group<material_data.get_n_groups(); group++)   //loop through all groups
    		if(conv[group] > conv_tol)
  		  {  	
    			std::cout<<" Solving for Group #"<<group<<std::endl;
    			conv[group]=0;
    			double den=0;
    			dofs = 0; 
    			bool suppress_source = false;
    			
    			reinit_phi_even_odd(group);  //reinitialize phi_even and phi_odd container 
    			
    			material_data_new = material_data;  //restore original cross-sections for top level
    			if(material_data.get_n_moments()!=1)
    				material_data_new.correct_scat_moments(group);  //get optimal corrected scattering cross-sections to start AMG with
    
    				
    			for (unsigned int N = material_data.get_n_moments(); N>=2||(material_data.get_n_moments()==1&&N==1); N=N/2)  //Go through the Hierarchy of Angular Multi-Grid
    			{
    				cout<<"Solving S"<<N<<" Equations"<<endl;
    				
    				//================ Get Info from Previous Grid ==================
    				for (unsigned moment=0; moment<N; moment=moment+2)
    					if (N == material_data.get_n_moments()/2)		//if top gird, solution from previous SI iteration were used 
    																											// to compute the correction to phi_even;
    					{
    						phi_even_new[group][moment] -= phi_even_old[group][moment];
    						phi_even_old[group][moment] = phi_even_new[group][moment];
    					}
    					else		//otherwise, no knowledge of previous iteration
    						phi_even_old[group][moment] = phi_even_new[group][moment];
    						
    				for (unsigned moment=1; moment<N; moment=moment+2)
    					if (N == material_data.get_n_moments()/2)		//if top gird, solution from previous SI iteration were used 
    																											// to compute the correction to phi_odd;
    					{
    						phi_odd_new[group][moment] -= phi_odd_old[group][moment];
    						phi_odd_old[group][moment] = phi_odd_new[group][moment];
    					}
    					else		//otherwise, no knowledge of previous iteration
    						phi_odd_old[group][moment] = phi_odd_new[group][moment];
    						
    				material_data_old = material_data_new;
    				
    				//================ Angular Quadrature ====================
    				compute_angular_quadrature(N, Pnm, mu, wt);
    				
    				//================ XS Correction =========================
    				material_data_new.set_n_moments(N);  //set new number of moments
    				if(material_data.get_n_moments()!=1)
    					material_data_new.correct_scat_moments(group);  //make optimal correction to the scattering moment (\sigma_s* = \sigma_s - (\sigma_{N/2} + \sigma_{N-1})/2 )
    							
    				for (unsigned int m=0; m< double(N/2.0); m++) // we make a loop over all the moments 
    				{	
    					//============== Solve for I+ =======================
    					std::cout<<"We begin to solve the I+ equation : m : "<<m<<std::endl;
    					spn_group[group]->setup_system (spn_group, m); // we create the matrix and the right-hand-side 
    					if ( N<material_data.get_n_moments() )
    						suppress_source = true;
    					assemble_system(m, group, suppress_source); // we give the values of the matrix and the right-hand-side
    	
    					spn_group[group]->solve(parameters.n_iteration_cg, parameters.convergence_tolerance,
    										spn_group[group]->solution_moment[m]); // we solve the system that we created just before					
    					
    					if(N == material_data.get_n_moments())
    					{
    						dofs = dofs + spn_group[group]->n_dofs;
    						std::cout<<"The number of dofs is :"<<spn_group[group]->n_dofs<<"\n";
    					}
    					cout<<"L2[psi(0)] = "<<spn_group[group]->solution_moment[m].l2_norm()<<endl;  //debug
    				}
    				//============== Solve for Phi_odd ========================	
    				std::cout<<"We begin to solve the I- equation : "<<std::endl;
    				compute_phi_odd(group, N, suppress_source);
    	
    				std::cout<<"The number of degrees of freedom for the direct problem is : "<<dofs<<std::endl;
    				
    				//================ Update phi_even ================	
    				compute_phi_even(group, N);
    				
    				//================ DSA =========================
    				if(N==2 || !AMG)  //apply P1_SA to the S2 equation, equivalent to solving the diffusion equation corresponding to the S2 problem
    				{
    					if(material_data.get_n_moments() == 2 || !AMG)
    					{
    						phi_old[group] = phi_even_old[group][0];
    						J_old[group] = phi_odd_old[group][1];
    					}
    					else
    					{
    						phi_old[group].reinit(spn_group[group]->n_dofs,false);
    						J_old[group].reinit(spn_group[group]->triangulation.n_active_cells()*spn_group[group]->fe.dofs_per_cell,false);
    					}
    					
    					
    					cout<<"Performing P1_SA Calculation"<<endl;
    					dsa->run(phi_old[group], phi_even_new[group][0], J_old[group], phi_odd_new[group][1], 
    					         parameters.n_iteration_cg, parameters.convergence_tolerance, group);
    				}
    				//================ Phi Correction =========================
    				for (unsigned int moment=0; moment<N; moment=moment+2)
    					phi_even[group][moment] += phi_even_new[group][moment];
    					
    				for (unsigned int moment=1; moment<N; moment=moment+2)
    					phi_odd[group][moment] += phi_odd_new[group][moment];
    					
    			  //================ Check if using Angular Multi-Grid ===============
    			  if(!AMG)
    			  	N = 0; //terminate angular grid hierachy if not using AMG
    				
    			}
    			for(unsigned int m=0; m<material_data.get_n_moments(); m=m+2)
    			  phi_even_new[group][m] = phi_even[group][m];
    			for(unsigned int m=1; m<material_data.get_n_moments(); m=m+2)
    			  phi_odd_new[group][m] = phi_odd[group][m];
    			
    			//============ Check Convergence ===================
    				if(j!=0)
    				{
    					for(unsigned int k=0; k<material_data.get_n_moments(); k=k+2)		// compute the diff in even phi moments
    					{
    						Vector<double> temp = phi_even[group][k];
    						temp -= phi_even_previous[group][k];
    						conv[group] += temp.l2_norm();
    						
    						den += phi_even[group][k].l2_norm();
    					}
    					for(unsigned int k=1; k<material_data.get_n_moments(); k=k+2)		// compute the diff in odd phi moments;
    					{
    						Vector<double> temp = phi_odd[group][k];
    						temp -= phi_odd_previous[group][k];
    						conv[group] += temp.l2_norm();
    						
    						den += phi_odd[group][k].l2_norm();
    					}
    				}	
    				else conv[group]=10;	
    				
    				conv[group] = conv[group]/den;
    				cout<<"den = "<<den<<endl;  //debug
    	cout<<"conv("<<group<<") = "<<conv[group]<<endl;  //debug
    	
    	for(unsigned int m=0; m<material_data.get_n_moments(); m=m+2)
    			phi_even_previous[group][m] = phi_even[group][m];
    	for(unsigned int m=1; m<material_data.get_n_moments(); m=m+2)
    			phi_odd_previous[group][m] = phi_odd[group][m]; 
    			
    			if(conv[group] > conv_tol)
    				not_converged = true;   //if any group didn't converge, repeat the source iteration process
    		}
    		else
    		{
    			std::cout<<"Group #"<<group<<" has converged!!"<<std::endl;
    		}
  				
  	}
  		
    for(unsigned int group = 0; group<material_data.get_n_groups(); group++)   //loop through all groups
		{
  		solution[group].reinit(spn_group[group]->n_dofs);	//here we create the vector which will contain the solution and 
  		
  		solution[group] = phi_even[group][0];
  		compute_Leakage(group);
  		check_conservation(group);
	  }	
	  
    if(dofs <= 1e5 || all_output)
  		output (cycle);		// we make the output for the solution
  	std::cout<<"time = "<<timer()<<std::endl<<std::endl;
	}	
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int main()
{	// this function calls the others to get the parameters and call the run function
	deallog.depth_console (0);

	Timer timer_total;
	timer_total.start();
	
	std::string filename;
	filename = "project_case1_mg.prm";
		
	const unsigned int dim=Dimension;

	ParameterHandler parameter_handler;
    SPN<dim>::Parameters parameters;
    parameters.declare_parameters (parameter_handler);
    parameter_handler.read_input (filename);
    parameters.get_parameters (parameter_handler);

    SPN<dim> calcul (parameters);
	calcul.run();
	
	std::cout<<"time elapsed = "<<timer_total()<<std::endl<<std::endl;
	
	return 0;
}
