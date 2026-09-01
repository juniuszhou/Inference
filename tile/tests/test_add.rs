use cutile::prelude::*;
use tile::kernels;

#[test]
fn test_add() {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (z, _, _) = kernels::add_1d(z, x, y).sync().expect("add_1d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![2.0; 1024]);
}

#[test]
fn test_add_with_pad() {
    let x = api::ones::<f32>(&[1023]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (z, _, _) = kernels::add_1d(z, x, y).sync().expect("add_1d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");

    let mut result = vec![2.0; 1024];
    result[1023] = 1.0;
    assert_eq!(z_host, result);
}

#[test]
fn test_add_1d_2() {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (z, _, _) = kernels::add_1d_2(z, x, y).sync().expect("add_1d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![2.0; 1024]);
}
