import linger.LINGER_tr as LINGER_tr
import shared_variables

# Refines the bulk model by further training it on the single-cell data
print(f'\nBeginning LINGER single cell training...')
LINGER_tr.training_cpu(
  shared_variables.bulk_model_dir,
  shared_variables.method,
  shared_variables.output_dir,
  shared_variables.activef,
  'Human'
  )

print(f'FINISHED TRAINING')